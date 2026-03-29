"""FastAPI application for SNOMED CT concept mapping via ADARV and FAISS search."""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import asyncio
import uvicorn
import logging
import re
import wordninja

from SnomedProcessor import SNOMEDProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

processor = None
thread_executor = None


class SNOMEDOutput(BaseModel):
    """Response model for a single SNOMED concept match."""
    conceptid: str
    conceptid_name: str
    variable_description: str
    category: str
    fhir_resource: str
    match_score: int
    source: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle.

    Initializes the SNOMEDProcessor (with GPU fallback to CPU) and a thread
    pool executor on startup, and shuts them down on application exit.
    """
    global processor, thread_executor
    logger.info("Starting SNOMED Mapping API...")
    thread_executor = ThreadPoolExecutor(max_workers=20)

    def _load_processor():
        try:
            return SNOMEDProcessor(use_gpu=True)
        except Exception:
            logger.warning("GPU unavailable. Falling back to CPU.")
            return SNOMEDProcessor(use_gpu=False)

    loop = asyncio.get_event_loop()
    processor = await loop.run_in_executor(thread_executor, _load_processor)
    yield
    logger.info("Shutting down SNOMED Mapping API...")
    if thread_executor:
        thread_executor.shutdown(wait=True)


app = FastAPI(
    title="SNOMED Mapping API",
    description="SNOMED CT concept search using ADARV and FAISS",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def split_with_wordninja(name: str) -> str:
    """Split a concatenated string into space-separated words using wordninja.

    Args:
        name: Concatenated or camelCase string.

    Returns:
        Space-separated string of detected words.
    """
    return " ".join(wordninja.split(name))


@app.get("/map_text", response_model=List[List[SNOMEDOutput]])
async def map_text(input: str = Query(..., description="Text to map, e.g., 'symptom first fever'"),
                   limit: int = Query(5, ge=1, le=5, description="Number of result groups to return (1-5)")):
    """Map input text to SNOMED CT concepts.

    Performs a two-stage search:
    1. Full-input PGVector similarity search against the ADARV data dictionary.
    2. Entity extraction followed by per-entity FAISS and PGVector searches.

    Results are deduplicated, prioritized by ADARV high-confidence matches,
    and sorted by average match score.

    Args:
        input: Free-text query string.
        limit: Maximum number of result groups to return (1-5).

    Returns:
        List of result groups, each containing a list of SNOMEDOutput matches.
    """
    input = input.strip()

    if not input:
        return []

    input = processor.clean_column_name(input)

    loop = asyncio.get_event_loop()
    results = []

    adarv_results = await loop.run_in_executor(thread_executor, processor.pgvector_top_match, input, 0.00)

    if adarv_results:
        for adarv_result in adarv_results:
            fhir = adarv_result["fhir_resource"] or "observation"
            cat = adarv_result["category"] or "Status"
            raw_snomed = adarv_result["snomed_codes"]
            adarv_score = int(round(adarv_result.get("similarity", 0) * 100))

            snomed_entries = []

            for part in raw_snomed.split(","):
                part = part.strip()

                code, name, description = None, None, "N/A"

                desc_match = re.search(r'\[(.*?)\]', part)
                if desc_match:
                    description = desc_match.group(1).strip()
                    part = part[:desc_match.start()].strip()

                match = re.match(r"(.+?)\s*\(\s*(\d{5,})\s*\)", part)
                if match:
                    name = match.group(1).strip()
                    code = match.group(2).strip()
                elif re.match(r"^(\d{5,})\s+(.+)$", part):
                    match_alt = re.match(r"^(\d{5,})\s+(.+)$", part)
                    code = match_alt.group(1).strip()
                    name = match_alt.group(2).strip()
                elif re.match(r"^(\d{5,})\s*\(\s*(.+?)\s*\)$", part):
                    match_alt2 = re.match(r"^(\d{5,})\s*\(\s*(.+?)\s*\)$", part)
                    code = match_alt2.group(1).strip()
                    name = match_alt2.group(2).strip()
                elif re.match(r"(.+?)\s+(\d{5,})\s*\)", part):
                    match_alt3 = re.match(r"(.+?)\s+(\d{5,})\s*\)", part)
                    name = match_alt3.group(1).strip()
                    code = match_alt3.group(2).strip()

                if code and name:
                    snomed_entries.append({
                        "conceptid": code,
                        "conceptid_name": name,
                        "variable_description": description,
                        "category": cat,
                        "fhir_resource": fhir,
                        "match_score": adarv_score,
                        "source": "ADARV"
                    })
                else:
                    logger.warning(f"Could not parse SNOMED string: '{part}'")

            results.append(snomed_entries)

    input = split_with_wordninja(input)

    entities = await loop.run_in_executor(thread_executor, processor.extract_entities_from_text, input)

    all_entity_matches = []
    for entity in entities:
        snomed_future = loop.run_in_executor(thread_executor, processor.search_snomed_faiss, entity, 20, 0.0)
        adarv_future = loop.run_in_executor(thread_executor, processor.pgvector_top_match, entity, 0.0)
        snomed_matches, adarv_match = await asyncio.gather(snomed_future, adarv_future)

        if adarv_match:
            adarv_match = adarv_match[0]
            fhir = adarv_match["fhir_resource"] or "observation"
            cat = adarv_match["category"] or "Status"

            time_variables = ["date", "time", "timestamp", "dt", "datetime", "Date/Time", "DateTime"]
            for time_variable in time_variables:
                if time_variable in input:
                    cat = "Date/Time"

            id_variables = ["number", "num"]
            for id_variable in id_variables:
                if id_variable in input:
                    cat = "ID"

            entity_matches = []
            if snomed_matches:
                for match in snomed_matches:
                    name = processor.preferred_terms.get(match["conceptId"], match["term"])
                    description = match.get("description", "N/A")

                    entity_matches.append({
                        "entity": entity,
                        "conceptid": match["conceptId"],
                        "conceptid_name": name,
                        "variable_description": description,
                        "category": cat,
                        "fhir_resource": fhir,
                        "match_score": int(round(match["similarity"] * 100)),
                        "source": "SNOMED"
                    })

            all_entity_matches.append(entity_matches)

    max_matches = max(len(matches) for matches in all_entity_matches) if all_entity_matches else 0

    for rank in range(max_matches):
        rank_group = []
        for entity_matches in all_entity_matches:
            if rank < len(entity_matches):
                rank_group.append(entity_matches[rank])

        if rank_group:
            results.append(rank_group)

    unique_results = []
    seen_conceptid_sets = set()

    for inner_list in results:
        conceptids = tuple(sorted(item['conceptid'] for item in inner_list))

        if conceptids not in seen_conceptid_sets:
            seen_conceptid_sets.add(conceptids)
            unique_results.append(inner_list)

    def calculate_priority_and_score(inner_list):
        """Calculate sorting key based on ADARV priority and average match score."""
        if not inner_list:
            return (0, 0)

        avg_score = sum(item['match_score'] for item in inner_list) / len(inner_list)

        has_high_adarv = any(
            item['source'] == 'ADARV' and item['match_score'] > 90
            for item in inner_list
        )

        priority = 1 if has_high_adarv else 0
        return (priority, avg_score)

    unique_results.sort(key=calculate_priority_and_score, reverse=True)
    return unique_results[:limit]


@app.get("/health")
def health_check():
    """Return API health status."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, timeout_keep_alive=30)
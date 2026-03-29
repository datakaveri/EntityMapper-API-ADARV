"""SNOMED CT processor for entity extraction and concept search using FAISS and PGVector."""

import os
import pickle
import numpy as np
import faiss
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
import spacy
from spacy.matcher import PhraseMatcher
from nltk.corpus import stopwords
import nltk
import re
from rapidfuzz import fuzz
import gc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.data.path.append("snomed_data/nltk_data")
STOPWORDS = set(stopwords.words("english"))

EMB_CACHE = 'snomed_data/snomed_terms.pkl'
INDEX_FILE = 'snomed_data/snomed_faiss.index'

DB_CONFIG = {
    'dbname': 'postgresdb',
    'user': 'postgres',
    'password': 'timescaledbpg',
    'host': '65.0.127.208',
    'port': 32588,
}


def clean_name_for_fuzzy(name):
    """Clean a SNOMED term name for fuzzy matching.

    Removes parenthesized text, leading N/n-digit prefixes,
    and normalizes to lowercase.

    Args:
        name: Raw SNOMED term name string.

    Returns:
        Cleaned, lowercase string suitable for fuzzy comparison.
    """
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = re.sub(r'^[Nn]\d+', '', name)
    return name.strip().lower()


class SNOMEDProcessor:
    """Handles SNOMED CT concept loading, entity extraction, and similarity search.

    Loads precomputed embeddings and a FAISS index for fast nearest-neighbor
    search, and uses SciSpacy for biomedical entity extraction from free text.
    Also supports PGVector-based similarity search against an ADARV data dictionary.
    """

    def __init__(self, use_gpu=True):
        """Initialize the SNOMED processor.

        Args:
            use_gpu: Whether to attempt GPU acceleration for FAISS and the
                     sentence transformer model.
        """
        self.model = None
        self.index = None
        self.terms = None
        self.term_concept = None
        self.preferred_terms = {}
        self.child_to_parent = {}
        self.snomed_terms_list = []
        self.matcher = None
        self.use_gpu = use_gpu
        self.nlp = None
        self._db_pool = None
        self._load_resources()

    def _load_resources(self):
        """Load SNOMED cache, FAISS index, sentence transformer, and SciSpacy model."""
        if not (os.path.exists(EMB_CACHE) and os.path.exists(INDEX_FILE)):
            raise ValueError(f"Missing SNOMED cache '{EMB_CACHE}' or index file.")

        logger.info("Loading cached SNOMED metadata and FAISS index...")

        with open(EMB_CACHE, 'rb') as f:
            cache = pickle.load(f)
            self.terms = cache['terms']
            self.term_concept = cache['concepts']
            self.preferred_terms = cache.get('conceptid_to_fsn', cache.get('preferred_terms', {}))
            self.child_to_parent = cache.get('child_to_parent', {})

        del cache
        gc.collect()

        device = 'cuda' if self.use_gpu else 'cpu'
        try:
            self.model = SentenceTransformer('snomed_data/sapbert', device=device)
            if not self.use_gpu:
                self.model.half()
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer: {e}")
            raise

        try:
            self.index = faiss.read_index(INDEX_FILE)
            if self.use_gpu and faiss.get_num_gpus() > 0:
                logger.info("Moving FAISS index to GPU...")
                res = faiss.StandardGpuResources()
                res.setDefaultNullStreamAllDevices()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        except Exception as e:
            logger.warning(f"FAISS GPU setup failed, using CPU: {e}")
            self.use_gpu = False

        try:
            self.nlp = spacy.load("en_core_sci_lg", disable=['parser', 'tagger', 'lemmatizer', 'attribute_ruler'])
            self.nlp.max_length = 1000000
        except Exception as e:
            logger.error(f"Failed to load SciSpacy model: {e}")
            raise ValueError("SciSpacy model `en_core_sci_lg` is not available.")

        self._prepare_snomed_terms()
        gc.collect()

    def clean_column_name(self, col: str) -> str:
        """Clean a column name by removing coded prefixes and inserting spaces.

        Args:
            col: Raw column name (e.g. 'Q00SymptomFirstFever').

        Returns:
            Human-readable string (e.g. 'Symptom First Fever').
        """
        cleaned = re.sub(r'^[A-Z]+\d+(?:_\d+)?', '', col)
        cleaned = cleaned.strip("_ ")
        cleaned = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', cleaned)
        return cleaned

    def _prepare_snomed_terms(self):
        """Prepare the SNOMED term list and spaCy PhraseMatcher patterns."""
        batch_size = 10000
        terms_processed = []

        for i in range(0, len(self.terms), batch_size):
            batch = self.terms[i:i+batch_size]
            batch_processed = [
                str(t).strip().lower()
                for t in batch
                if t and len(str(t).strip()) > 1
            ]
            terms_processed.extend(batch_processed)

            if i % (batch_size * 5) == 0:
                gc.collect()

        self.snomed_terms_list = list(set(terms_processed))
        del terms_processed

        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        batch_size = 1000
        for i in range(0, len(self.snomed_terms_list), batch_size):
            batch_terms = self.snomed_terms_list[i:i+batch_size]
            try:
                patterns = [self.nlp.make_doc(term) for term in batch_terms]
                self.matcher.add(f"SNOMED_TERMS_{i}", patterns)
            except Exception as e:
                logger.warning(f"Failed to add pattern batch {i}: {e}")

        logger.info(f"Prepared {len(self.snomed_terms_list)} SNOMED terms.")
        gc.collect()

    def _remove_substring_entities(self, entities):
        """Filter out entities that are word-level subsets of longer entities.

        Args:
            entities: List of entity strings.

        Returns:
            Filtered list with redundant substring entities removed.
        """
        if not entities:
            return entities

        sorted_entities = sorted(entities, key=len, reverse=True)
        filtered_entities = []

        for entity in sorted_entities:
            entity_words = set(entity.lower().split())
            is_subset = False
            for preserved_entity in filtered_entities:
                preserved_words = set(preserved_entity.lower().split())
                if entity_words.issubset(preserved_words) and entity_words != preserved_words:
                    is_subset = True
                    break
            if not is_subset:
                filtered_entities.append(entity)
        return filtered_entities

    def extract_entities_from_text(self, text):
        """Extract biomedical entities from free text using SciSpacy and PhraseMatcher.

        Combines NER entities with phrase-matched SNOMED terms, removes stopwords
        and single-character tokens, and deduplicates substring overlaps.

        Args:
            text: Input text string (truncated to 1000 chars).

        Returns:
            List of deduplicated entity strings.
        """
        if len(text) > 1000:
            text = text[:1000]

        doc = self.nlp(text)

        matches = self.matcher(doc)
        phrase_spans = [doc[start:end] for _, start, end in matches]
        phrase_texts = {span.text.strip().lower() for span in phrase_spans}

        all_tokens = set()
        for span in phrase_spans:
            all_tokens.update(
                t.text.strip().lower()
                for t in span if not t.is_stop and t.is_alpha and len(t.text.strip()) > 1
            )

        ents = []
        for ent in doc.ents:
            tokens = [
                t.text.strip().lower() for t in ent
                if not t.is_stop and t.is_alpha and len(t.text.strip()) > 1
            ]
            filtered = [tok for tok in tokens if tok not in all_tokens]
            if filtered:
                ents.append(" ".join(filtered))

        combined = list(set(ents) | phrase_texts)
        combined = [c for c in combined if len(c) > 1 and c not in STOPWORDS and not c.isdigit()]

        deduplicated = self._remove_substring_entities(combined)
        return deduplicated

    def pgvector_top_match(self, text, threshold=0.00, topn=5):
        """Find the top matching ADARV entries using PGVector cosine similarity.

        Args:
            text: Query text to encode and search.
            threshold: Minimum similarity score (default 0.00).
            topn: Number of results to return (default 5).

        Returns:
            List of matching rows as dicts, or None on failure.
        """
        try:
            emb = self.model.encode([text], convert_to_numpy=True).astype('float32')
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute(f"""
                SELECT snomed_codes, fhir_resource, category,
                    (1 - (embeddings <=> %s::vector)) AS similarity
                FROM adarv_data_dict_sapbert
                ORDER BY similarity DESC
                LIMIT {topn};
            """, (emb[0].tolist(),))

            rows = cur.fetchall()

        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return None
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

        return rows

    def search_snomed_faiss(self, text, topk=10, threshold=0.00):
        """Search for SNOMED concepts using FAISS nearest-neighbor lookup.

        Args:
            text: Query text to encode.
            topk: Number of nearest neighbors to retrieve (default 10).
            threshold: Minimum cosine similarity to include (default 0.00).

        Returns:
            List of result dicts with term, conceptId, similarity, and description.
        """
        try:
            q_emb = self.model.encode([text], convert_to_numpy=True).astype('float32')
            faiss.normalize_L2(q_emb)
            sims, idxs = self.index.search(q_emb, topk)

            results = []
            for i in range(topk):
                idx = idxs[0][i]
                sim = float(sims[0][i])
                if sim >= threshold:
                    concept_id = self.term_concept[idx]
                    parent_id = self.child_to_parent.get(concept_id)
                    description = "N/A"
                    if parent_id:
                        description = self.preferred_terms.get(parent_id, "N/A")

                    results.append({
                        'term': self.terms[idx],
                        'conceptId': self.term_concept[idx],
                        'similarity': sim,
                        'description': description
                    })
            return results
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    def __del__(self):
        """Release database connections and run garbage collection."""
        if hasattr(self, '_db_pool') and self._db_pool:
            self._db_pool.closeall()
        gc.collect()
# EntityMapper-API-ADARV

FastAPI service that maps free-text clinical variables to SNOMED CT codes. Uses FAISS for nearest-neighbor search and PGVector for similarity search against the ADARV data dictionary.

## Setup

The `snomed_data/` directory must contain the following before running:

- `snomed_terms.pkl` — SNOMED term metadata (terms, concept IDs, preferred names, parent mappings)
- `snomed_faiss.index` — FAISS index of SNOMED term embeddings
- `sapbert/` — SapBERT model directory (`config.json`, `pytorch_model.bin`, tokenizer files)
- `nltk_data/` — NLTK data with `corpora/stopwords`

These files are large and not tracked by Git. Get them from shared storage or generate using the embedding pipeline.

## Run

```bash
pip install -r requirements.txt
pip install https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

uvicorn SnomedSearch:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -t snomed-search .
docker run -p 8000:8000 snomed-search
```

## Endpoints

- `GET /map_text?input=<text>&limit=<n>` — Returns up to `limit` (1–5) ranked SNOMED match groups for the given text.
- `GET /health` — Health check.

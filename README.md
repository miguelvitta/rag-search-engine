# rag-search-engine

Movie search from the command line: keyword search (BM25), semantic search (embeddings), and hybrid search (weighted or Reciprocal Rank Fusion) with optional LLM query enhancement and re-ranking.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- An [OpenRouter](https://openrouter.ai/) API key, for the hybrid search and evaluation commands.

## Setup

1. Install the dependencies (the first run also downloads PyTorch, which is large):

   ```bash
   uv sync
   ```

2. Add the data files. `data/` is not in the repository; create it at the project root with:

   | File | Content |
   | --- | --- |
   | `data/movies.json` | `{"movies": [{"id": 1, "title": "...", "description": "..."}]}` |
   | `data/stopwords.txt` | One stopword per line |
   | `data/golden_dataset.json` | `{"test_cases": [{"query": "...", "relevant_docs": ["<movie title>"]}]}` (only for evaluation) |

3. Create a `.env` file at the project root with your key:

   ```
   OPENROUTER_API_KEY=<your key>
   ```

   The hybrid search and evaluation CLIs fail at startup without it, even for commands that don't call the LLM.

## Usage

Run every command from the project root. Indexes and embeddings are written to `cache/`; delete it to rebuild them.

### Keyword search

Build the index first:

```bash
uv run cli/keyword_search_cli.py build
uv run cli/keyword_search_cli.py bm25search "space adventure"
```

Other commands: `search`, `tf`, `idf`, `tfidf`, `bm25idf`, `bm25tf`.

### Semantic search

Embeddings are created on first use. The first run downloads the `all-MiniLM-L6-v2` model.

```bash
uv run cli/semantic_search_cli.py search "space adventure" --limit 5
uv run cli/semantic_search_cli.py search_chunked "space adventure" --limit 5
```

Other commands: `verify`, `embed_text`, `verify_embeddings`, `embed_query`, `chunk`, `semantic_chunk`, `embed_chunks`.

### Hybrid search

The first run downloads the `cross-encoder/ms-marco-TinyBERT-L2-v2` re-ranking model.

```bash
uv run cli/hybrid_search_cli.py weighted-search "space adventure" --alpha 0.5 --limit 5
uv run cli/hybrid_search_cli.py rrf-search "space adventure" -k 60 --limit 5
```

`rrf-search` options:

- `--enhance spell|expand|rewrite`: rewrite the query with the LLM before searching.
- `--rerank-method individual|batch|cross_encoder`: re-rank the results (`individual` and `batch` use the LLM; `cross_encoder` runs locally).

### Evaluation

Precision@k, recall@k and F1 against `data/golden_dataset.json`:

```bash
uv run cli/evaluation_cli.py --limit 5
```

### Check the LLM connection

```bash
uv run cli/test_llm.py
```

Every CLI prints its commands and options with `--help`.

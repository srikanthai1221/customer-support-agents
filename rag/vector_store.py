"""FAISS vector store: one index per agent's knowledge collection."""
import json
import re
import sys

import faiss
import numpy as np
from openai import OpenAI

from config import COLLECTIONS, EMBED_MODEL, INDEX_DIR, MIN_SCORE, OPENAI_API_KEY, TOP_K

_client = OpenAI(api_key=OPENAI_API_KEY)
_cache: dict = {}


def _embed(texts: list[str]) -> np.ndarray:
    """Turn texts into normalised vectors (inner product = cosine similarity)."""
    response = _client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def _chunk_markdown(text: str, source: str) -> list[dict]:
    """Split a markdown file into one chunk per '## ' section."""
    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else source
    chunks = []
    for section in re.split(r"\n(?=## )", text):
        section = section.strip()
        if not section or (section.startswith("# ") and "\n" not in section):
            continue  # skip empty or title-only pieces
        body = section if section.startswith("# ") else f"{title}\n{section}"
        chunks.append({"source": source, "text": body})
    return chunks


def build_index(name: str) -> int:
    """Embed every .md file of one collection and save a FAISS index."""
    chunks = []
    for path in sorted(COLLECTIONS[name].glob("*.md")):
        chunks.extend(_chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    if not chunks:
        raise ValueError(f"No .md files found in {COLLECTIONS[name]}")

    vectors = _embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    out_dir = INDEX_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _cache.pop(name, None)
    return len(chunks)


def build_all() -> None:
    for name in COLLECTIONS:
        print(f"{name}: {build_index(name)} chunks indexed")


def _load(name: str):
    """Load an index once per process; build it first if it doesn't exist."""
    if name not in _cache:
        out_dir = INDEX_DIR / name
        if not (out_dir / "index.faiss").exists():
            build_index(name)
        index = faiss.read_index(str(out_dir / "index.faiss"))
        chunks = json.loads((out_dir / "chunks.json").read_text(encoding="utf-8"))
        _cache[name] = (index, chunks)
    return _cache[name]


def search(name: str, query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE) -> list[dict]:
    """Return the best-matching chunks above the similarity cut-off."""
    index, chunks = _load(name)
    scores, ids = index.search(_embed([query]), min(top_k, index.ntotal))
    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0 and score >= min_score:
            hits.append({**chunks[idx], "score": round(float(score), 3)})
    return hits


def format_hits(hits: list[dict]) -> str:
    """Text an agent can read: each passage with its source file."""
    if not hits:
        return "NO_MATCH: no relevant passage found."
    return "\n\n".join(f"[{h['source']} | score {h['score']}]\n{h['text']}" for h in hits)


TEST_QUERIES = [
    "Where is my order? It was supposed to arrive yesterday.",
    "I want to return the shoes I bought last week. How do I get a refund?",
    "I forgot my password and can't log in.",
    "The blender I received is broken. Can I get a replacement?",
    "You charged my card three times! Fix this now.",
    "Do you sell gift cards for a company of 500 employees with custom branding?",
]

if __name__ == "__main__":
    if "--test" in sys.argv:
        for q in TEST_QUERIES:
            top = search("policies", q, min_score=0.0)[0]
            print(f"{top['score']:.3f}  {top['source']:<26} {q}")
    else:
        build_all()
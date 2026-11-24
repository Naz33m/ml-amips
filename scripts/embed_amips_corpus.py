"""Embed the prepared AMIPS JSONL corpus and draft queries."""

import argparse
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_parser() -> argparse.ArgumentParser:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=pathlib.Path,
        default=root / "data" / "cleaned" / "amips_corpus.jsonl",
    )
    parser.add_argument(
        "--queries",
        type=pathlib.Path,
        default=root / "data" / "cleaned" / "amips_queries_draft.jsonl",
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default=root / "data" / "cleaned" / "amips_smoke_embeddings",
    )
    parser.add_argument("--encoder", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    return parser


def main(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    corpus = _read_jsonl(args.corpus)
    queries = _read_jsonl(args.queries)
    if not corpus or not queries:
        raise ValueError("Corpus and query JSONL files must both be non-empty")

    key_ids = [record.get("id") for record in corpus]
    if any(not isinstance(record_id, str) or not record_id for record_id in key_ids):
        raise ValueError("Every corpus record must have a non-empty string 'id'")
    if len(key_ids) != len(set(key_ids)):
        raise ValueError("Corpus record IDs must be unique")

    key_texts = [record.get("search_text") for record in corpus]
    query_texts = [record.get("query") for record in queries]
    if any(not isinstance(text, str) or not text.strip() for text in key_texts):
        raise ValueError("Every corpus record must have non-empty 'search_text'")
    if any(not isinstance(text, str) or not text.strip() for text in query_texts):
        raise ValueError("Every query record must have non-empty 'query'")

    known_ids = set(key_ids)
    for index, query in enumerate(queries):
        candidates = query.get("candidate_record_ids")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Query {index} must have candidate_record_ids")
        unknown = set(candidates) - known_ids
        if unknown:
            raise ValueError(f"Query {index} references unknown record IDs: {unknown}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.encoder, device=args.device)

    def encode(texts: list[str]) -> np.ndarray:
        vectors = model.encode(
            texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("Encoder returned invalid embedding values")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Encoder returned embeddings that are not normalized")
        return vectors

    key_vectors = encode(key_texts)
    query_vectors = encode(query_texts)
    if key_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("Query and key embedding dimensions do not match")

    args.output_path.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path / "keys.npy", key_vectors)
    np.save(args.output_path / "queries.npy", query_vectors)
    (args.output_path / "key_ids.json").write_text(
        json.dumps(key_ids, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    query_map = [
        {
            "query_index": index,
            "query": record["query"],
            "candidate_record_ids": record["candidate_record_ids"],
            "query_status": record.get("query_status"),
            "kind": record.get("kind"),
        }
        for index, record in enumerate(queries)
    ]
    (args.output_path / "query_map.json").write_text(
        json.dumps(query_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "encoder": args.encoder,
        "normalized_embeddings": True,
        "key_count": len(key_vectors),
        "query_count": len(query_vectors),
        "dimension": int(key_vectors.shape[1]),
        "corpus_sha256": _sha256(args.corpus),
        "queries_sha256": _sha256(args.queries),
        "outputs": ["keys.npy", "queries.npy", "key_ids.json", "query_map.json"],
        "note": (
            "Draft queries and candidate IDs are for smoke testing only, not "
            "validated relevance labels."
        ),
    }
    (args.output_path / "embedding_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(key_vectors)} keys and {len(query_vectors)} queries "
        f"with dimension {key_vectors.shape[1]} to {args.output_path}"
    )


if __name__ == "__main__":
    main(_get_parser().parse_args())

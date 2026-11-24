"""Prepare a focused, safety-filtered AMIPS corpus and draft retrieval queries."""

import argparse
import hashlib
import json
import pathlib
import re
from typing import Any


INCLUDED_KINDS = {
    "herbal_heritage",
    "herb",
    "remedy",
    "condiment",
    "medicinal_plant",
}
EXCLUDED_KINDS = {"traditional_treatment", "research_trial"}
PROCEDURE_PATTERN = re.compile(
    r"(?i)\b(?:infusion|decoction|poultice|boil|steep|soak|chew|"
    r"drink|apply|mix|prepare|preparation|take on an? empty stomach|"
    r"warm gently|serve in)\b"
)
AMOUNT_PATTERN = re.compile(
    r"(?i)\b\d+(?:\s*[-–]\s*\d+)?\s*(?:mg|g|ml|tsp|tbsp|cups?|"
    r"drops?|leaves?|threads?|cloves?|pieces?)\b|\b\d+\s*:\s*\d+\b|"
    r"\b\d+(?:\.\d+)?\s*%"
)
INGREDIENT_AMOUNT_PREFIX = re.compile(
    r"(?i)^\s*(?:optional:\s*)?(?:\d+(?:\s*/\s*\d+)?|a few)\s*"
    r"(?:[-–]\s*\d+\s*)?"
)
INGREDIENT_MEASURE_PREFIX = re.compile(
    r"(?i)^\s*(?:teaspoons?|tablespoons?|tsp|tbsp|cups?|drops?|"
    r"pinch(?:es)?|glass(?:es)?|ml|mg|g)\s*"
)
INGREDIENT_ACTION = re.compile(
    r"(?i)\b(?:take|drink|apply|boil|steep|soak|chew|mix|prepare|"
    r"warm gently|serve|burn|add to|use as)\b"
)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sanitize_text(value: str, stats: dict[str, int]) -> str:
    retained = []
    for sentence in re.split(r"(?<=[.!?])\s+", _clean(value)):
        if not sentence:
            continue
        if PROCEDURE_PATTERN.search(sentence):
            stats["procedure_sentences_removed"] += 1
            continue
        sentence, n = AMOUNT_PATTERN.subn(" ", sentence)
        stats["amount_patterns_removed"] += n
        sentence = _clean(sentence)
        if sentence:
            retained.append(sentence)
    return " ".join(retained)


def _sanitize_ingredient(value: str, stats: dict[str, int]) -> str:
    text = _clean(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+\s*/\s*", text) or INGREDIENT_ACTION.search(text):
        stats["ingredient_instructions_removed"] += 1
        return ""
    text = re.sub(r"(?i)^\s*optional:\s*add\s+", "", text)
    text = re.sub(r"(?i)\bthumb-sized piece\b", "", text)
    text = INGREDIENT_AMOUNT_PREFIX.sub("", text)
    text = INGREDIENT_MEASURE_PREFIX.sub("", text)
    text, n = AMOUNT_PATTERN.subn(" ", text)
    stats["amount_patterns_removed"] += n
    text = re.sub(r"(?i)^\s*optional:\s*", "", text)
    text = re.sub(r"(?i)^\s*of\s+", "", text)
    return _clean(text.strip(" ,;:-"))


def _display_name(record: dict[str, Any]) -> str:
    names = record.get("names") or {}
    english = names.get("english") or []
    return _clean(english[0]) if english else _clean(record.get("title", ""))


def _provenance(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("provenance") or {}


def _build_corpus_record(
    record: dict[str, Any], stats: dict[str, int]
) -> dict[str, Any]:
    kind = record["kind"]
    display_name = _display_name(record)
    description = _sanitize_text(record.get("description") or "", stats)
    traditional_uses = []
    for item in record.get("traditional_uses") or []:
        raw_text = item.get("text", "") if isinstance(item, dict) else str(item)
        text = _sanitize_text(raw_text, stats)
        if text:
            traditional_uses.append(
                {
                    "period": item.get("period") if isinstance(item, dict) else None,
                    "text": text,
                    "status": "reported_source_claim_unverified",
                }
            )
    benefit_claims = []
    for item in record.get("reported_benefits") or []:
        text = _sanitize_text(str(item), stats)
        if text:
            benefit_claims.append(
                {"text": text, "status": "unverified_source_claim"}
            )
    ingredients = []
    for item in record.get("ingredients") or []:
        text = _sanitize_ingredient(str(item), stats)
        if text:
            ingredients.append(text)

    caution = _clean(str(record.get("caution") or ""))
    names = record.get("names") or {}
    identity = [
        display_name,
        _clean(str(names.get("scientific") or "")),
        _clean(str(record.get("family") or "")),
        _clean(str(record.get("category") or "")),
        _clean(str(record.get("kind") or "")),
    ]
    identity = [part for part in identity if part]
    search_parts = identity + [description]
    search_parts.extend(item["text"] for item in traditional_uses)
    search_parts.extend(item["text"] for item in benefit_claims)
    search_parts.extend(ingredients)
    if caution:
        search_parts.append(f"Source caution: {caution}")
    label = (
        "Source-reported health content; claims unverified; not medical advice."
        if benefit_claims or traditional_uses
        else "Source-attributed heritage context; not treatment guidance."
    )

    return {
        "id": record["id"],
        "kind": kind,
        "title": _clean(str(record.get("title") or display_name)),
        "names": names,
        "family": record.get("family") or None,
        "category": record.get("category"),
        "description": description,
        "traditional_uses": traditional_uses,
        "reported_benefits": benefit_claims,
        "ingredients": ingredients,
        "caution_as_written": caution or None,
        "source_body_system_tags": record.get("body_systems") or [],
        "provenance": _provenance(record),
        "health_claim_status": (
            "unverified_source_claims"
            if benefit_claims or traditional_uses
            else "context_only"
        ),
        "safety_boundary": (
            "No diagnosis, treatment recommendation, dose, or preparation guidance. "
            "Source claims are not independently verified."
        ),
        "search_text": label + " " + " ".join(part for part in search_parts if part),
    }


def _draft_query(corpus_record: dict[str, Any]) -> dict[str, Any]:
    name = corpus_record["title"]
    kind = corpus_record["kind"]
    if kind == "medicinal_plant":
        question = (
            f"What plant identification and reported traditional-use information "
            f"does the archive contain for {name}?"
        )
    elif kind == "remedy":
        question = (
            f"What source-reported information is recorded for the remedy {name}? "
            "Distinguish claims from verified evidence; do not give preparation or dosage."
        )
    elif kind == "herbal_heritage":
        question = (
            f"What heritage context does the archive record for {name}, and what "
            "limitations does its source state?"
        )
    else:
        question = (
            f"What source-reported information is recorded about {name}, and what "
            "is its evidence status?"
        )
    return {
        "query": question,
        "candidate_record_ids": [corpus_record["id"]],
        "query_status": "draft_candidate_requires_human_review",
        "kind": corpus_record["kind"],
    }


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=pathlib.Path, required=True)
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=root / "data" / "cleaned"
    )
    args = parser.parse_args()

    input_bytes = args.input_json.read_bytes()
    source = json.loads(input_bytes.decode("utf-8-sig"))
    records = source.get("records")
    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a 'records' array")
    declared_total = source.get("counts", {}).get("total")
    if declared_total is not None and declared_total != len(records):
        raise ValueError(
            f"Input count mismatch: declared {declared_total}, found {len(records)}"
        )

    output_records = []
    excluded_counts: dict[str, int] = {}
    stats = {
        "procedure_sentences_removed": 0,
        "amount_patterns_removed": 0,
        "ingredient_instructions_removed": 0,
    }
    for record in records:
        kind = record.get("kind")
        if kind in EXCLUDED_KINDS:
            excluded_counts[kind] = excluded_counts.get(kind, 0) + 1
            continue
        if kind not in INCLUDED_KINDS:
            raise ValueError(f"Unexpected record kind: {kind!r}")
        output_records.append(_build_corpus_record(record, stats))

    ids = [record["id"] for record in output_records]
    if len(ids) != len(set(ids)):
        raise ValueError("Focused corpus contains duplicate record IDs")
    queries = [_draft_query(record) for record in output_records]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / "amips_corpus.jsonl"
    query_path = args.output_dir / "amips_queries_draft.jsonl"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in output_records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with query_path.open("w", encoding="utf-8", newline="\n") as stream:
        for query in queries:
            stream.write(json.dumps(query, ensure_ascii=False, sort_keys=True) + "\n")

    counts: dict[str, int] = {}
    for record in output_records:
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1
    manifest = {
        "input_file": args.input_json.name,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "input_record_count": len(records),
        "included_record_count": len(output_records),
        "included_by_kind": counts,
        "excluded_by_kind": excluded_counts,
        "draft_query_count": len(queries),
        "draft_query_policy": (
            "One safe, name-based question per included record. Candidate IDs are "
            "suggestions, not validated relevance labels."
        ),
        "text_filter_stats": stats,
        "outputs": [corpus_path.name, query_path.name],
        "notes": [
            "The 19 traditional-treatment records and 7 unverified study records were excluded.",
            "Ingredient amounts and preparation instructions were omitted from searchable text.",
            "Claims remain source-reported and unverified; review the query candidates before evaluation.",
            "This prepares text inputs only; it does not create embeddings or train a model.",
        ],
    }
    manifest_path = args.output_dir / "amips_input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(output_records)} corpus records and {len(queries)} draft queries."
    )


if __name__ == "__main__":
    main()

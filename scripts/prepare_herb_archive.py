"""Normalize the source CSVs into a provenance-preserving conversation archive."""

import argparse
import csv
import hashlib
import json
import pathlib
import re
from typing import Any


SOURCE_FILES = {
    "herbs (1).csv": "herb",
    "ancient_remedies (1).csv": "remedy",
    "condiments (1).csv": "condiment",
    "scientific_trials (1).csv": "study_record",
}
LIST_FIELDS = {"benefits", "ingredients", "traditions"}
SCHEMA_VERSION = 1
EVIDENCE_NOTE = (
    "Source text has not been independently fact-checked. No supporting citations "
    "were present in the supplied records. This archive is not medical advice."
)


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _parse_list(value: str | None) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    try:
        values = next(csv.reader([text], skipinitialspace=True))
    except csv.Error as exc:
        raise ValueError(f"Unable to parse list field {value!r}") from exc
    return [item.strip() for item in values if item.strip()]


def _read_source(path: pathlib.Path, record_type: str) -> tuple[list[dict[str, Any]], str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(name is None for name in reader.fieldnames):
            raise ValueError(f"{path.name}: missing or invalid CSV header")
        records = []
        seen_ids = set()
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path.name}:{row_number}: unexpected extra columns")
            source_id = _clean_text(row.get("id"))
            name = _clean_text(row.get("name") or row.get("title"))
            if not source_id or not name:
                raise ValueError(f"{path.name}:{row_number}: missing id or name/title")
            if source_id in seen_ids:
                raise ValueError(f"{path.name}:{row_number}: duplicate id {source_id!r}")
            seen_ids.add(source_id)

            source_fields: dict[str, Any] = {}
            for field, value in row.items():
                if field in LIST_FIELDS:
                    source_fields[field] = _parse_list(value)
                else:
                    source_fields[field] = _clean_text(value)

            benefit_claims = [
                {"text": claim, "evidence_status": "unverified_source_claim"}
                for claim in source_fields.get("benefits", [])
            ]
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "id": f"{record_type}:{source_id}",
                "record_type": record_type,
                "name": name,
                "scientific_name": source_fields.get("scientific_name") or None,
                "category": source_fields.get("category") or None,
                "summary": source_fields.get("description") or None,
                "traditional_use": (
                    source_fields.get("ancient_use")
                    or source_fields.get("traditions")
                    or None
                ),
                "modern_use": source_fields.get("modern_use") or None,
                "ingredients": source_fields.get("ingredients", []),
                "preparation": source_fields.get("preparation") or None,
                "benefit_claims": benefit_claims,
                "safety_notes": [
                    note
                    for note in (
                        source_fields.get("caution"),
                        source_fields.get("safety_ar"),
                    )
                    if note
                ],
                "evidence_note": EVIDENCE_NOTE,
                "source": {
                    "file": path.name,
                    "row": row_number,
                    "record_id": source_id,
                },
                "source_fields": source_fields,
            }
            if record_type == "study_record":
                record["evidence_status"] = "unverified_study_record"
                record["evidence_warning"] = (
                    "The source provides no citation, study identifier, date, or "
                    "sample size. Outcomes and dosage text are unverified claims."
                )
            text_parts = [f"Record: {name}"]
            excluded_search_fields = {
                "id",
                "name",
                "title",
                "emoji",
                "emoji2",
                "color",
                "image",
                "is_archived",
                "is_featured",
            }
            for field, value in source_fields.items():
                if field in excluded_search_fields or not value:
                    continue
                if field == "benefits":
                    text_parts.extend(
                        f"Unverified source claim: {claim['text']}"
                        for claim in benefit_claims
                    )
                elif isinstance(value, list):
                    text_parts.append(f"{field}: {'; '.join(value)}")
                else:
                    text_parts.append(f"{field}: {value}")
            record["search_text"] = (
                "Unverified source content; not medical advice. "
                + " ".join(text_parts)
            )
            records.append(record)
    return records, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument("--input-dir", type=pathlib.Path, default=root / "data")
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=root / "data" / "cleaned"
    )
    args = parser.parse_args()

    all_records: list[dict[str, Any]] = []
    manifest_sources = []
    for filename, record_type in SOURCE_FILES.items():
        path = args.input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required source CSV not found: {path}")
        records, digest = _read_source(path, record_type)
        all_records.extend(records)
        manifest_sources.append(
            {
                "file": filename,
                "record_type": record_type,
                "record_count": len(records),
                "sha256": digest,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "herbal_archive.jsonl"
    with archive_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in all_records:
            stream.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(all_records),
        "record_type_counts": {
            record_type: sum(r["record_type"] == record_type for r in all_records)
            for record_type in sorted(set(SOURCE_FILES.values()))
        },
        "sources": manifest_sources,
        "evidence_note": EVIDENCE_NOTE,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(all_records)} records to {archive_path}")


if __name__ == "__main__":
    main()

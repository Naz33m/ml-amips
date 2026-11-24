"""Extract herb, remedy, and related heritage records from a Knowledge_bank export."""

import argparse
import hashlib
import json
import pathlib
import re
import unicodedata
from typing import Any


PROFILE_FILES = {
    "heritage_herbs.md": "materia_medica_profile",
    "heritage_condiments.md": "food_and_spice_profile",
}
ARCHIVE_NOTE = (
    "This is an extraction of supplied source material, not medical advice or "
    "independent verification of its health claims."
)
CONTEXT_PATTERN = re.compile(
    r"(?i)\b(?:herb\w*|plant\w*|botan\w*|ethnobotan\w*|"
    r"medicinal[- ]material\w*|home[- ]remed\w*|traditional remed\w*|"
    r"miswak|arak|sidr|ghaf|date[- ]palm|phoenix dactylifera|"
    r"salvadora persica|saffron|anise|ginger|chamomile|thyme|aloe)\b"
)
INDEX_HEADING = re.compile(
    r"(?i)^(?:what the heritage records say|the records on|"
    r"everything the heritage knowledge covers|the whole collection|"
    r"all the remedies|all the scientific studies|all the healing condiments|"
    r"the herbs, spices and foods)"
)


def _read_sections(path: pathlib.Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    sections: list[tuple[str, list[str]]] = []
    title = ""
    body: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            if title:
                sections.append((title, body))
            title = match.group(1).strip()
            body = [line]
        elif title:
            body.append(line)
    if title:
        sections.append((title, body))
    return [(title, "\n".join(lines).strip()) for title, lines in sections]


def _slug(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    )
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", ascii_value.lower())).strip(
        "-"
    )


def _bullet_block(text: str, label: str) -> list[str]:
    lines = text.splitlines()
    target = label.casefold()
    for index, line in enumerate(lines):
        if line.strip().rstrip(":").casefold() == target.rstrip(":"):
            result = []
            for candidate in lines[index + 1 :]:
                candidate = candidate.strip()
                if not candidate:
                    if result:
                        break
                    continue
                if not candidate.startswith("- "):
                    break
                result.append(candidate[2:].strip())
            return result
    return []


def _line_value(text: str, label: str) -> str | None:
    pattern = re.compile(rf"(?im)^\s*{re.escape(label)}:\s*(.+?)\s*$")
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _record_id(source_path: str, title: str, ordinal: int) -> str:
    path_slug = _slug(pathlib.PurePosixPath(source_path).with_suffix("").as_posix())
    title_slug = _slug(title) or f"section-{ordinal}"
    return f"knowledge-bank:{path_slug}:{title_slug}"


def _base_record(
    *,
    source_path: str,
    title: str,
    ordinal: int,
    record_type: str,
    source_text: str,
    status: str,
) -> dict[str, Any]:
    return {
        "id": _record_id(source_path, title, ordinal),
        "record_type": record_type,
        "name": title,
        "source_status": status,
        "source": {"file": source_path, "section": title},
        "source_text": source_text,
        "archive_note": ARCHIVE_NOTE,
    }


def _extract_profile(
    source_path: str,
    title: str,
    text: str,
    ordinal: int,
    record_type: str,
) -> dict[str, Any]:
    record = _base_record(
        source_path=source_path,
        title=title,
        ordinal=ordinal,
        record_type=record_type,
        source_text=text,
        status="current_source",
    )
    scientific_line = re.search(r"(?im)^Scientific name:\s*(.+)$", text)
    if scientific_line:
        line = scientific_line.group(1)
        record["scientific_name"] = re.split(r"\.\s+Kind:", line, maxsplit=1)[0].strip(
            " ."
        ) or None
        kind = re.search(r"\bKind:\s*(.+?)(?:\.\s*Known as:|$)", line)
        known_as = re.search(r"\bKnown as:\s*(.+?)\.?\s*$", line)
        if kind:
            record["kind"] = kind.group(1).strip()
        if known_as:
            record["also_known_as"] = known_as.group(1).strip()
    record["benefit_claims"] = [
        {"text": value, "status": "unverified_source_claim"}
        for value in _bullet_block(text, "Benefits reported")
    ]
    record["preparation_claims"] = _bullet_block(text, "How it is prepared")
    record["safety_notes_as_written"] = _line_value(text, "Caution")
    record["health_claim_status"] = "unverified_source_claims"
    record["search_text"] = (
        f"Unverified source claims; not medical advice. {title}. {text}"
    )
    return record


def _extract_remedy(
    source_path: str, title: str, text: str, ordinal: int
) -> dict[str, Any]:
    record = _base_record(
        source_path=source_path,
        title=title,
        ordinal=ordinal,
        record_type="traditional_remedy",
        source_text=text,
        status="current_source",
    )
    origin_line = re.search(
        r"(?im)^\s*Origin:\s*(.+?)(?:\.\s*Tradition:|$)", text
    )
    tradition = re.search(r"(?im)\bTradition:\s*(.+?)\s*$", text)
    record["origin"] = origin_line.group(1).strip() if origin_line else None
    record["tradition"] = tradition.group(1).strip().rstrip(" .") if tradition else None
    record["ingredients_as_written"] = _bullet_block(text, "What goes in it")
    record["benefit_claims"] = [
        {"text": value, "status": "unverified_source_claim"}
        for value in _bullet_block(text, "Known for")
    ]
    record["health_claim_status"] = "unverified_source_claims"
    record["preparation_status"] = "source_preserved_not_safety_reviewed"
    incomplete = [
        item
        for item in record["ingredients_as_written"]
        if re.fullmatch(r"\d+\s*/\s*", item)
    ]
    if incomplete:
        record["source_quality_flags"] = [
            "incomplete_or_truncated_ingredient_entries"
        ]
        record["incomplete_ingredient_entries"] = incomplete
    record["search_text"] = (
        f"Unverified source recipe; not medical advice. {title}. {text}"
    )
    return record


def _extract_trial(source_path: str, title: str, text: str, ordinal: int) -> dict[str, Any]:
    record = _base_record(
        source_path=source_path,
        title=title,
        ordinal=ordinal,
        record_type="unverified_study_record",
        source_text=text,
        status="current_source",
    )
    record["evidence_status"] = "unverified_study_record"
    record["missing_evidence_fields"] = [
        "supporting citation or study identifier",
        "sample size",
        "publication details",
    ]
    record["search_text"] = (
        f"Unverified study claim; not established clinical evidence. {title}. {text}"
    )
    return record


def _extract_context(
    source_path: str, title: str, text: str, ordinal: int
) -> dict[str, Any]:
    record = _base_record(
        source_path=source_path,
        title=title,
        ordinal=ordinal,
        record_type="heritage_context",
        source_text=text,
        status="current_source",
    )
    record["evidence_status"] = "source_context_not_clinical_evidence"
    record["confidence_as_written"] = _line_value(text, "How sure the sources are")
    record["boundary_as_written"] = _line_value(text, "Boundary when speaking of it")
    record["search_text"] = (
        f"Heritage and research context; not treatment guidance. {title}. {text}"
    )
    return record


def _extract_archived(
    source_path: str, title: str, text: str, ordinal: int
) -> dict[str, Any]:
    record = _base_record(
        source_path=source_path,
        title=title,
        ordinal=ordinal,
        record_type="archived_herb_context",
        source_text=text,
        status="removed_archive",
    )
    record["ingest_policy"] = "exclude_by_default"
    record["evidence_status"] = "archived_context_not_clinical_evidence"
    record["search_text"] = (
        f"Archived, removed source; exclude by default. {title}. {text}"
    )
    return record


def _append_sections(
    records: list[dict[str, Any]],
    path: pathlib.Path,
    relative_path: str,
    predicate: Any,
    extractor: Any,
) -> None:
    for ordinal, (title, text) in enumerate(_read_sections(path), start=1):
        if predicate(title, text):
            records.append(extractor(relative_path, title, text, ordinal))


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knowledge-bank-dir",
        type=pathlib.Path,
        default=root / "data" / "knowledge_bank_2026-11-23",
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=root / "data" / "cleaned"
    )
    args = parser.parse_args()
    source_root = args.knowledge_bank_dir / "source_files"
    records: list[dict[str, Any]] = []
    sources: dict[str, str] = {}

    def path_for(relative: str) -> pathlib.Path:
        path = source_root / pathlib.Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Required source file not found: {path}")
        source_name = "source_files/" + pathlib.Path(relative).as_posix()
        sources[source_name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    for filename, record_type in PROFILE_FILES.items():
        relative = f"source_files/{filename}"
        path = path_for(filename)
        _append_sections(
            records,
            path,
            relative,
            lambda title, text: bool(re.search(r"(?im)^Benefits reported:", text)),
            lambda src, title, text, ordinal: _extract_profile(
                src, title, text, ordinal, record_type
            ),
        )

    remedies_path = path_for("heritage_remedies.md")
    _append_sections(
        records,
        remedies_path,
        "source_files/heritage_remedies.md",
        lambda title, text: "What goes in it:" in text,
        _extract_remedy,
    )

    trials_path = path_for("heritage_trials.md")
    _append_sections(
        records,
        trials_path,
        "source_files/heritage_trials.md",
        lambda title, text: bool(re.search(r"(?im)^Purpose:", text)),
        _extract_trial,
    )

    om_saif_dir = source_root / "om-saif"
    for path in sorted(om_saif_dir.glob("*.md")):
        relative = "source_files/" + path.relative_to(source_root).as_posix()
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        _append_sections(
            records,
            path,
            relative,
            lambda title, text: bool(CONTEXT_PATTERN.search(title + "\n" + text))
            and not INDEX_HEADING.search(title),
            _extract_context,
        )

    archived_relative = (
        "source_files/_removed_2026-11-22/herb_collection_um_saif.md"
    )
    archived_file = archived_relative.removeprefix("source_files/")
    archived_path = path_for(archived_file)
    _append_sections(
        records,
        archived_path,
        archived_relative,
        lambda title, text: bool(re.search(r"(?im)^Names:", text)),
        _extract_archived,
    )

    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Generated duplicate record IDs")

    counts: dict[str, int] = {}
    for record in records:
        counts[record["record_type"]] = counts.get(record["record_type"], 0) + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "knowledge_bank_herbs_remedies.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "record_count": len(records),
        "record_type_counts": counts,
        "source_sha256": sources,
        "output": output_path.name,
        "notes": [
            ARCHIVE_NOTE,
            "Source files were used; reconstructed documents_text copies were not duplicated.",
            "The explicitly removed herb collection is included separately and marked exclude_by_default.",
            "Incomplete recipe text was preserved and flagged; no missing amounts were inferred.",
        ],
    }
    manifest_path = args.output_dir / "knowledge_bank_extract_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} records to {output_path}")
    print("Record types:", json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()

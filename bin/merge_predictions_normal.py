#!/usr/bin/env python3
"""Merge ``predictions_normal.csv`` files from every class folder under one experiment root.

This matches the evaluation layout produced by ``train_eval_mvtec_classes.sh``:

    {experiment_dir}/{mvtec_class}/predictions_normal.csv

Each per-class file has columns ``ID``, ``Label`` (see ``patchcore.utils.plot_segmentation_images``).
The merged file adds a ``Class`` column so IDs that repeat across classes (e.g. ``000``)
stay unique.

Example::

    python bin/merge_predictions_normal.py evaluated_results/p01 \\
        -o evaluated_results/p01/merged_predictions_normal.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


DEFAULT_INPUT_NAME = "predictions_normal.csv"
MERGED_DEFAULT_NAME = "merged_predictions_normal.csv"


def list_class_prediction_csvs(experiment_dir: Path, csv_name: str) -> list[tuple[str, Path]]:
    """Return sorted ``(class_name, csv_path)`` for direct child dirs containing ``csv_name``."""

    pairs: list[tuple[str, Path]] = []
    for child in sorted(experiment_dir.iterdir()):
        if not child.is_dir():
            continue
        path = child / csv_name
        if path.is_file():
            pairs.append((child.name, path))
    return pairs


def read_predictions_csv(csv_path: Path, class_name: str | None) -> list[tuple[str | None, str, str]]:
    """Parse ID and Label rows; optionally associate with *class_name*."""

    rows: list[tuple[str | None, str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return rows
        normalized = [h.strip() for h in header]
        if normalized[:2] != ["ID", "Label"]:
            raise ValueError(
                f"{csv_path}: expected header ID, Label - got {header!r}"
            )
        for row in reader:
            if len(row) < 2:
                continue
            rid, label = row[0].strip(), row[1]
            rows.append((class_name, rid, label))
    return rows


def merge_normal_predictions(
    experiment_dir: Path,
    output_path: Path,
    csv_name: str,
    include_class_column: bool,
) -> int:
    """Merge per-class normal CSVs; returns exit code (0 ok, 1 errors)."""

    merged_rows: list[tuple[str | None, str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    found_any = False

    if not experiment_dir.is_dir():
        print(f"error: not a directory: {experiment_dir}", file=sys.stderr)
        return 1

    for class_name, csv_path in list_class_prediction_csvs(experiment_dir, csv_name):
        found_any = True
        cls_for_rows = class_name if include_class_column else None
        try:
            block = read_predictions_csv(csv_path, cls_for_rows)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        for cls, rid, label in block:
            key = (cls or "", rid)
            if key in seen_keys:
                print(
                    f"warning: duplicate row ({cls!r}, {rid!r}) - skipping extra from {csv_path}",
                    file=sys.stderr,
                )
                continue
            seen_keys.add(key)
            merged_rows.append((cls, rid, label))

    if not found_any:
        print(
            f"error: no `{csv_name}` files found under class subdirectories of {experiment_dir}",
            file=sys.stderr,
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        if include_class_column:
            writer.writerow(["Class", "ID", "Label"])
            for cls, rid, label in merged_rows:
                writer.writerow([cls, rid, label])
        else:
            writer.writerow(["ID", "Label"])
            for _, rid, label in merged_rows:
                writer.writerow([rid, label])

    print(
        f"Wrote {len(merged_rows)} rows from class folders under {experiment_dir} -> {output_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge predictions_normal.csv from each immediate subdirectory "
            "of an experiment folder (one MVTec class per subdirectory)."
        )
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help=(
            "Experiment root containing class subfolders "
            "(e.g. evaluated_results/p01)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            f"Output CSV path (default: <experiment_dir>/{MERGED_DEFAULT_NAME})."
        ),
    )
    parser.add_argument(
        "--csv-name",
        default=DEFAULT_INPUT_NAME,
        help=f"Filename to merge in each class folder (default: {DEFAULT_INPUT_NAME}).",
    )
    parser.add_argument(
        "--no-class-column",
        action="store_true",
        help="Keep only ID and Label columns (error if duplicate IDs appear).",
    )
    args = parser.parse_args(argv)

    experiment_dir = args.experiment_dir.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else experiment_dir / MERGED_DEFAULT_NAME
    )

    include_class = not args.no_class_column
    if not include_class:
        # Preflight: detect duplicate IDs across files.
        all_ids: list[str] = []
        csv_name = args.csv_name
        for _, path in list_class_prediction_csvs(experiment_dir, csv_name):
            rows = read_predictions_csv(path, None)
            all_ids.extend(r[1] for r in rows)
        if len(all_ids) != len(set(all_ids)):
            print(
                "error: duplicate IDs across classes - omit --no-class-column "
                "or fix inputs.",
                file=sys.stderr,
            )
            return 1

    return merge_normal_predictions(
        experiment_dir,
        output_path,
        args.csv_name,
        include_class_column=include_class,
    )


if __name__ == "__main__":
    raise SystemExit(main())

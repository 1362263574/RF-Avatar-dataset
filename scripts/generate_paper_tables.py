# -*- coding: utf-8 -*-
"""Generate paper-ready RF-Avatar dataset tables from dataset_stats/."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready dataset tables from dataset_stats.")
    parser.add_argument("--stats-dir", required=True, help="Existing dataset_stats directory.")
    parser.add_argument("--output-dir", required=True, help="Output paper_tables directory.")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text: Any) -> str:
    value = str(text)
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def router_config_id(index: int) -> str:
    return f"RC{index}"


def pretty_scene_name(name: str) -> str:
    mapping = {"lab": "Lab", "newlab": "NewLab", "room": "Room"}
    return mapping.get(name.lower(), name)


def pretty_occluder(name: str) -> str:
    if not name:
        return name
    return name[0].upper() + name[1:]


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def build_canonical_action_mapping(action_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    mappings: List[Dict[str, Any]] = []
    for row in action_rows:
        raw = row["action_name"]
        suggested = raw
        modifier = ""
        confidence = "high"
        note = ""
        if raw.endswith("_occlusion_full"):
            suggested = raw[: -len("_occlusion_full")]
            modifier = "full"
        elif raw.endswith("_occlusion_lower"):
            suggested = raw[: -len("_occlusion_lower")]
            modifier = "lower"
        elif raw.endswith("_occlusion_upper"):
            suggested = raw[: -len("_occlusion_upper")]
            modifier = "upper"
        elif raw in {"occlusion_full", "occlusion_lower", "occlusion_upper"}:
            suggested = "occlusion"
            modifier = raw.split("_", 1)[1]
            confidence = "medium"
            note = "Base action is ambiguous from the raw name alone."
        elif raw == "newlab_nothing":
            suggested = "nothing"
            confidence = "medium"
            note = "Background-only physical-occlusion calibration clip."
        if suggested != raw or modifier or note:
            mappings.append(
                {
                    "raw_action": raw,
                    "suggested_canonical_action": suggested,
                    "occlusion_modifier": modifier,
                    "confidence": confidence,
                    "note": note,
                }
            )
    return mappings


def main() -> int:
    args = parse_args()
    stats_dir = Path(args.stats_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_json(stats_dir / "dataset_summary.json")
    dataset_summary_md = (stats_dir / "dataset_summary.md").read_text(encoding="utf-8")
    source_latex = (stats_dir / "latex_tables.md").read_text(encoding="utf-8")
    subject_action = read_csv(stats_dir / "subject_action_stats.csv")
    action_stats = read_csv(stats_dir / "action_stats.csv")
    people_distribution = read_csv(stats_dir / "people_distribution.csv")
    scene_router = read_csv(stats_dir / "scene_router_stats.csv")
    occlusion_free = read_csv(stats_dir / "occlusion_free_stats.csv")
    physical = read_csv(stats_dir / "physical_occlusion_stats.csv")
    quality = read_csv(stats_dir / "quality_control_stats.csv")
    split_stats = read_csv(stats_dir / "split_stats.csv")
    split_suggestion = read_json(stats_dir / "split_suggestion.json")

    counts = summary["counts"]
    parsing_notes = summary.get("parsing_notes", [])
    manual_confirmations = summary.get("manual_confirmations", [])

    all_modality_reason = (
        "All-modality valid sample count is not directly available from the current summary files."
    )

    # Table II
    mp_rows: List[List[Any]] = []
    total_frames = 0
    total_instances = 0
    for row in people_distribution:
        n = int(row["num_people"])
        frames = int(row["frames"])
        instances = int(row["human_instances"])
        total_frames += frames
        total_instances += instances
        mp_rows.append([f"{n}-person", frames, instances])
    mp_rows.append(["Total", total_frames, total_instances])

    # Table III
    router_rows = []
    for idx, row in enumerate(scene_router, start=1):
        sessions = to_int(row["sessions"])
        background_sessions = to_int(row.get("background_sessions", 0))
        valid_frames = to_int(row["valid_frames"])
        human_instances = to_int(row["human_instances"])
        if valid_frames > 0 and human_instances > 0:
            router_type = "Human"
        elif sessions == 1 and valid_frames == 0 and human_instances == 0:
            router_type = "Background"
        else:
            router_type = "No retained human samples"
        router_rows.append(
            {
                "scene": pretty_scene_name(row["scene_name"]),
                "router_position": row["router_position"],
                "router_config_id": router_config_id(idx),
                "sessions": sessions,
                "background_sessions": background_sessions,
                "valid_frames": valid_frames,
                "human_instances": human_instances,
                "type": router_type,
                "tx_coords": row["tx_coords"],
                "rx_coords": row["rx_coords"],
            }
        )
    background_router_count = sum(1 for row in router_rows if row["type"] == "Background")
    human_router_count = sum(1 for row in router_rows if row["type"] == "Human")
    other_router_count = sum(1 for row in router_rows if row["type"] == "No retained human samples")
    background_only_found = background_router_count > 0

    # Table IV
    phys_rows = []
    phys_total_sessions = 0
    phys_total_raw = 0
    phys_total_valid = 0
    phys_total_discarded = 0
    for row in physical:
        sessions = int(row["sessions"]) if row["sessions"].isdigit() else row["sessions"]
        raw = int(row["raw_paired_frames"]) if row["raw_paired_frames"].isdigit() else row["raw_paired_frames"]
        valid = int(row["valid_paired_frames"]) if row["valid_paired_frames"].isdigit() else row["valid_paired_frames"]
        discarded = int(row["discarded_frames"]) if row["discarded_frames"].isdigit() else row["discarded_frames"]
        phys_rows.append([pretty_occluder(row["occluder_type"]), f"{row['distance_cm']} cm", sessions, raw, valid, discarded])
        if isinstance(sessions, int):
            phys_total_sessions += sessions
        if isinstance(raw, int):
            phys_total_raw += raw
        if isinstance(valid, int):
            phys_total_valid += valid
        if isinstance(discarded, int):
            phys_total_discarded += discarded
    phys_rows.append(["Total", "-", phys_total_sessions, phys_total_raw, phys_total_valid, phys_total_discarded])

    # Table I
    table_i_rows = [
        ("Subjects", counts["number_of_subjects"]),
        ("Action categories", counts["number_of_action_categories"]),
        ("Indoor scenes", counts["number_of_indoor_scenes"]),
        ("Recorded router deployment configurations", counts["number_of_router_deployment_configurations"]),
        ("Human reconstruction router configurations", human_router_count),
        ("Background/no-person router configurations", background_router_count),
        ("Recording sessions", counts["number_of_recording_sessions"]),
        ("Valid RGB-D frames", counts["number_of_valid_rgbd_frames"]),
        ("Valid CSI samples", counts["number_of_valid_csi_samples"]),
        ("Skeleton-valid frames", counts["number_of_valid_body38_skeleton_annotations"]),
        ("Point-cloud-supervised samples", counts["number_of_valid_human_point_cloud_pseudo_labels"]),
        ("Human instances", counts["number_of_human_instances"]),
        ("Maximum people per frame", counts["maximum_number_of_people_per_frame"]),
        ("Physical-occlusion paired samples", phys_total_valid),
    ]
    if other_router_count > 0:
        table_i_rows.insert(
            6,
            ("Additional recorded configurations without retained multimodal human samples", other_router_count),
        )

    # Table V
    eval_rows = [
        ["Standard reconstruction", "Occlusion-free multimodal data", "Clean test sessions", "Evaluate basic multi-person 3D skeleton and point cloud reconstruction"],
        ["Multi-person reconstruction", "1--3 person data", "Test sessions grouped by the number of people", "Evaluate multi-person instance decoupling"],
        ["Synthetic visual occlusion", "Online CutOut-augmented RGB-D with unchanged CSI", "Masked RGB-D generated from clean test sessions", "Test robustness to visual missingness"],
        ["Physical occlusion", "Paired occluded/unoccluded recordings", "Real physical-occlusion sessions", "Evaluate robustness to real occluders and RF propagation shifts"],
        ["Cross-router-position generalization", "Selected router configurations", "Held-out or moved router configurations", "Evaluate deployment generalization with router coordinates"],
    ]

    # Table VI
    physical_pairs_by_split: Dict[str, int] = {}
    for split_name, records in split_suggestion.get("splits", {}).items():
        physical_pairs_by_split[split_name] = sum(
            int(record["valid_frames"]) for record in records if record.get("kind") == "physical_occlusion"
        )
    split_rows = []
    split_frame_sum = 0
    split_instance_sum = 0
    split_physical_sum = 0
    for row in split_stats:
        split = row["split"]
        if split not in {"train", "val", "test"}:
            continue
        split_frame_sum += int(row["frames"])
        split_instance_sum += int(row["human_instances"])
        split_physical_sum += int(physical_pairs_by_split.get(split, 0))
        split_rows.append(
            [
                split.title(),
                row["sessions"],
                row["frames"],
                "Not recomputed",
                row["human_instances"],
                physical_pairs_by_split.get(split, "Not directly available"),
            ]
        )
    split_inconsistent = split_frame_sum != counts["number_of_valid_rgbd_frames"] or split_instance_sum != counts["number_of_human_instances"]

    # Appendix A
    subject_agg: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"actions": set(), "sessions": 0, "frames": 0, "instances": 0})
    for row in subject_action:
        subject = row["subject_id"]
        subject_agg[subject]["actions"].add(row["action_name"])
        subject_agg[subject]["sessions"] += int(row["sessions"])
        subject_agg[subject]["frames"] += int(row["valid_frames"])
        subject_agg[subject]["instances"] += int(row["human_instances"])
    appendix_a_rows = []
    for subject in sorted(subject_agg):
        appendix_a_rows.append(
            [
                subject,
                len(subject_agg[subject]["actions"]),
                subject_agg[subject]["sessions"],
                subject_agg[subject]["frames"],
                subject_agg[subject]["instances"],
            ]
        )

    mapping_rows = build_canonical_action_mapping(action_stats)
    write_csv(
        output_dir / "canonical_action_mapping_suggestion.csv",
        mapping_rows,
        ["raw_action", "suggested_canonical_action", "occlusion_modifier", "confidence", "note"],
    )

    # Appendix B
    qc_lookup = {row["metric"]: row for row in quality}
    appendix_b_rows = [
        ["Raw samples", qc_lookup["total raw samples"]["value"]],
        ["Valid RGB-D frames", counts["number_of_valid_rgbd_frames"]],
        ["Valid CSI samples", counts["number_of_valid_csi_samples"]],
        ["Skeleton-valid frames", counts["number_of_valid_body38_skeleton_annotations"]],
        ["Valid point cloud pseudo-labels", counts["number_of_valid_human_point_cloud_pseudo_labels"]],
        ["Human instances", counts["number_of_human_instances"]],
        ["Missing RGB-D", qc_lookup["missing RGB-D"]["value"]],
        ["Invalid depth", qc_lookup["invalid depth"]["value"]],
        ["Invalid CSI", qc_lookup["invalid CSI"]["value"]],
        ["Missing skeleton", qc_lookup["missing skeleton"]["value"]],
        ["Incomplete skeleton", qc_lookup["incomplete skeleton"]["value"]],
        ["Ghost skeleton removed", qc_lookup["ghost skeleton removed"]["value"]],
        ["Invalid point cloud", qc_lookup["invalid point cloud"]["value"]],
        ["Shape error", qc_lookup["shape error samples"]["value"]],
        ["NaN / Inf samples", "Not checked" if qc_lookup["NaN / Inf samples"]["value"] == "unknown" else qc_lookup["NaN / Inf samples"]["value"]],
        ["Final retained samples", qc_lookup["valid samples after filtering"]["value"]],
    ]

    dataset_paragraph = (
        f"We collect a synchronized RGB-D and WiFi CSI dataset from {counts['number_of_subjects']} subjects performing "
        f"{counts['number_of_action_categories']} action categories across {counts['number_of_indoor_scenes']} indoor scenes "
        f"and {counts['number_of_router_deployment_configurations']} recorded router deployment configurations. "
        f"Among these router configurations, {human_router_count} contain valid human samples and {background_router_count} "
        f"{'is' if background_router_count == 1 else 'are'} used for background/no-person calibration. "
        + (
            f"The remaining {other_router_count} configuration{'s' if other_router_count != 1 else ''} currently contribute recorded sessions "
            "without retained multimodal human samples under the present filtering rules. "
            if other_router_count > 0
            else ""
        )
        + "The WiFi receiver is "
        "co-located with the ZED-X camera optical center and is therefore defined as the origin of the camera coordinate "
        "system, while the transmitter position is represented by a manually measured 3D offset relative to this "
        f"receiver/camera origin. The dataset contains {counts['number_of_valid_rgbd_frames']} valid RGB-D frames, "
        f"{counts['number_of_valid_csi_samples']} valid CSI samples, {counts['number_of_valid_body38_skeleton_annotations']} "
        f"skeleton-valid frames, and {counts['number_of_human_instances']} human instances, with up to "
        f"{counts['maximum_number_of_people_per_frame']} people appearing in one frame. After depth-based pseudo-label "
        f"generation and quality control, {counts['number_of_valid_human_point_cloud_pseudo_labels']} reliable human-centric "
        f"point cloud pseudo-labels are retained for point cloud supervision. In addition, we construct a paired "
        f"physical-occlusion subset containing {phys_total_valid} valid occluded--unoccluded paired samples with three "
        "occluder materials and three occluder distances."
    )

    synthetic_aug = (
        "Synthetic visual occlusion is not an independently collected dataset. During training, we apply online "
        "random CutOut to RGB and depth observations with probability 0.6. The mask area is randomly sampled from "
        "15% to 50% of the image area, and the aspect ratio is sampled from 0.3 to 3.3. The masked values are set "
        "to 0.0. The CSI stream is kept unchanged, since this augmentation only simulates visual missingness and does "
        "not model RF propagation changes caused by real physical occluders. For the synthetic occlusion stress test, "
        "we mask the lower 50% of RGB-D observations while keeping the original CSI unchanged. The synthetic mask only "
        "simulates visual missingness and does not simulate RF propagation shifts."
    )

    # Markdown output
    md_parts: List[str] = []
    md_parts.append("# RF-Avatar Paper Dataset Tables")
    md_parts.append("")
    md_parts.append("## Dataset Paragraph")
    md_parts.append("")
    md_parts.append(dataset_paragraph)
    md_parts.append("")
    md_parts.append("## Synthetic Visual Occlusion Augmentation")
    md_parts.append("")
    md_parts.append(synthetic_aug)
    md_parts.append("")
    md_parts.append("## Table I. Dataset Summary")
    md_parts.append("")
    md_parts.append(md_table(["Item", "Value"], [[a, b] for a, b in table_i_rows]))
    md_parts.append("")
    md_parts.append("## Table II. Multi-person Distribution")
    md_parts.append("")
    md_parts.append(md_table(["Number of people", "Frames", "Human instances"], mp_rows))
    md_parts.append("")
    md_parts.append("Note: The multi-person distribution is computed over skeleton-valid frames. The number of people per frame is determined by valid BODY-38 skeleton instances rather than directory-level subject tokens.")
    md_parts.append("")
    md_parts.append("## Table III. Scene and Router-position Distribution")
    md_parts.append("")
    md_parts.append(
        md_table(
            ["Scene", "Router position", "Config ID", "Sessions", "Valid frames", "Human instances", "Type"],
            [[r["scene"], r["router_position"], r["router_config_id"], r["sessions"], r["valid_frames"], r["human_instances"], r["type"]] for r in router_rows],
        )
    )
    md_parts.append("")
    md_parts.append(
        "Note: For all router configurations, the WiFi receiver is aligned with the ZED-X camera optical center and is treated as the camera coordinate origin. "
        "The transmitter position is represented by a manually measured offset relative to this receiver/camera origin. "
        "One recorded router configuration appears only in background/no-person sessions and is not counted as a human reconstruction configuration. "
        + (
            f"In addition, {other_router_count} recorded configuration{'s' if other_router_count != 1 else ''} contain recorded sessions but no retained multimodal human samples under the current filtering rules. "
            if other_router_count > 0
            else ""
        )
    )
    md_parts.append("")
    md_parts.append("## Table IV. Physical-occlusion Dataset")
    md_parts.append("")
    md_parts.append(md_table(["Occluder", "Distance", "Sessions", "Raw paired frames", "Valid paired frames", "Discarded frames"], phys_rows))
    md_parts.append("")
    md_parts.append("Note: Valid paired frames denote occluded--unoccluded pairs retained after quality control.")
    md_parts.append("")
    md_parts.append("## Table V. Evaluation Protocols")
    md_parts.append("")
    md_parts.append(md_table(["Protocol", "Training data", "Testing data", "Purpose"], eval_rows))
    md_parts.append("")
    md_parts.append("Note: Synthetic visual occlusion is not an independently collected dataset; it is generated online or during stress testing from clean RGB-D samples, while CSI remains unchanged.")
    md_parts.append("")
    md_parts.append("## Table VI. Train / Validation / Test Split (Draft / Not for Main Paper)")
    md_parts.append("")
    md_parts.append(md_table(["Split", "Sessions", "RGB-D frames", "Skeleton-valid frames", "Human instances", "Physical-occlusion pairs"], split_rows))
    md_parts.append("")
    md_parts.append("Draft note: Current split statistics have inconsistent human-instance totals and should not be used in the paper until recomputed. All splits should be performed at the recording-session level to avoid temporal leakage between training and testing.")
    md_parts.append("")
    md_parts.append("## Appendix Table A. Subject-level Data Distribution")
    md_parts.append("")
    md_parts.append(md_table(["Subject", "Action categories", "Sessions", "Valid frames", "Human instances"], appendix_a_rows))
    md_parts.append("")
    md_parts.append("Note: In multi-person sessions, a recording session may be attributed to multiple participating subjects; therefore, subject-level totals are not expected to sum to the dataset-level frame count. Raw action categories are parsed from directory or metadata names and may include occlusion-related suffixes.")
    md_parts.append("")
    md_parts.append("## Appendix Table B. Quality Control Statistics")
    md_parts.append("")
    md_parts.append(md_table(["Item", "Count / Status"], appendix_b_rows))
    md_parts.append("")
    write_text(output_dir / "paper_dataset_tables.md", "\n".join(md_parts) + "\n")

    # CSV summary md
    csv_summary = [
        "# CSV Summary",
        "",
        "## Source Files",
        "",
        f"- dataset_summary.json: {stats_dir / 'dataset_summary.json'}",
        f"- subject_action_stats.csv: {stats_dir / 'subject_action_stats.csv'}",
        f"- action_stats.csv: {stats_dir / 'action_stats.csv'}",
        f"- people_distribution.csv: {stats_dir / 'people_distribution.csv'}",
        f"- scene_router_stats.csv: {stats_dir / 'scene_router_stats.csv'}",
        f"- occlusion_free_stats.csv: {stats_dir / 'occlusion_free_stats.csv'}",
        f"- physical_occlusion_stats.csv: {stats_dir / 'physical_occlusion_stats.csv'}",
        f"- quality_control_stats.csv: {stats_dir / 'quality_control_stats.csv'}",
        f"- split_stats.csv: {stats_dir / 'split_stats.csv'}",
        f"- split_suggestion.json: {stats_dir / 'split_suggestion.json'}",
        "",
        "## Derived Checks",
        "",
        f"- Table II total frames = {total_frames}",
        f"- Table II total human instances = {total_instances}",
        f"- Table IV total valid paired frames = {phys_total_valid}",
        f"- Human reconstruction router configurations = {human_router_count}",
        f"- Background-only router configuration detected = {'Yes' if background_only_found else 'No'}",
        f"- Additional recorded configurations without retained multimodal human samples = {other_router_count}",
        f"- Suggested split physical-occlusion pairs = train {physical_pairs_by_split.get('train', 0)}, val {physical_pairs_by_split.get('val', 0)}, test {physical_pairs_by_split.get('test', 0)}",
        f"- Draft split RGB-D frame total = {split_frame_sum}",
        f"- Draft split human-instance total = {split_instance_sum}",
        f"- Draft split totals consistent with dataset summary = {'No' if split_inconsistent else 'Yes'}",
    ]
    write_text(output_dir / "paper_dataset_tables.csv_summary.md", "\n".join(csv_summary) + "\n")

    notes = [
        "# Paper Table Notes",
        "",
        "## Key Notes",
        "",
        f"- {all_modality_reason}",
        "- Table II uses the effective people count from the valid skeleton-derived distribution, not the raw directory subject token count.",
        "- One recorded router configuration appears only in background/no-person sessions and is not counted as a human reconstruction configuration unless explicitly stated.",
        f"- {other_router_count} recorded router configuration(s) currently contribute recorded sessions without retained multimodal human samples under the present filtering rules." if other_router_count > 0 else "- No additional zero-valid multimodal router configurations were detected beyond the explicit background calibration configuration.",
        "- The WiFi receiver is co-located with the ZED-X stereo camera optical center and is therefore defined as the origin of the camera coordinate system.",
        "- The transmitter position is represented by a manually measured 3D offset relative to the receiver/camera origin.",
        "- Since the receiver is aligned with the stereo camera optical center, the receiver coordinate is not separately stored in the metadata. The camera coordinate origin is used as the receiver coordinate.",
        "- The `nothing` and `newlab_nothing` clips are background calibration captures and should be interpreted as background/baseline measurements rather than human action categories.",
        "- All splits should be performed at the recording-session level to avoid temporal leakage between training and testing.",
        "- Raw action categories are parsed from directory or metadata names and may include occlusion-related suffixes.",
        "- The current dataset summary reports subject-count and skeleton-count mismatch frames; these require manual confirmation before writing a stronger claim about annotation completeness.",
        "- Quality-control statistics are kept for appendix or internal reporting because several fields such as ghost skeleton removal and NaN/Inf checks are unavailable or not checked.",
        "- Final retained samples in the QC table should not be interpreted as all-modality valid samples.",
        "",
        "## Detected Consistency Checks",
        "",
        f"- Physical-occlusion total valid paired frames from CSV = {phys_total_valid}.",
        "- The expected reference count from perfect_train_list.json is 1980.",
        f"- Match status = {'matched' if phys_total_valid == 1980 else 'mismatched'}.",
        f"- Background-only router configuration detected = {'Yes' if background_only_found else 'No'}.",
        f"- Additional recorded configurations without retained multimodal human samples = {other_router_count}.",
        f"- Subject-count and skeleton-count mismatch mentioned in summary = {'Yes' if any('mismatch' in note for note in manual_confirmations) else 'No'}.",
        f"- Draft split RGB-D frame total = {split_frame_sum} versus dataset valid RGB-D frames = {counts['number_of_valid_rgbd_frames']}.",
        f"- Draft split human-instance total = {split_instance_sum} versus dataset human instances = {counts['number_of_human_instances']}.",
        f"- Current split statistics have inconsistent human-instance totals and should not be used in the paper until recomputed." if split_inconsistent else "- Current split totals are internally consistent.",
        "",
        "## Source Annotations",
        "",
        "### Parsing Notes",
        *[f"- {note}" for note in parsing_notes],
        "",
        "### Manual Confirmations",
        *[
            f"- {note}"
            for note in manual_confirmations
            if "tx_antenna_offset_mm is acceptable" not in note
            and "background 'nothing' sessions" not in note
        ],
        "",
        "### Reused Stats Artifacts",
        "",
        "- dataset_summary.md and latex_tables.md were read as existing references but the paper_tables outputs were freshly regenerated from the structured CSV/JSON stats.",
        "- The previous latex_tables.md was not edited in place.",
    ]
    write_text(output_dir / "paper_table_notes.md", "\n".join(notes) + "\n")

    # LaTeX
    tex: List[str] = []
    tex.append("% RF-Avatar paper-ready dataset tables")
    tex.append("% Generated from dataset_stats/")
    tex.append("")
    tex.append("% Dataset paragraph")
    tex.append(dataset_paragraph)
    tex.append("")
    tex.append("% Synthetic visual occlusion augmentation")
    tex.append(synthetic_aug)
    tex.append("")
    tex.append("\\begin{table}[t]")
    tex.append("\\caption{RF-Avatar dataset summary used in the experimental setup.}")
    tex.append("\\label{tab:dataset_summary}")
    tex.append("\\centering")
    tex.append("\\resizebox{\\columnwidth}{!}{%")
    tex.append("\\begin{tabular}{lr}")
    tex.append("\\hline")
    tex.append("Item & Value \\\\")
    tex.append("\\hline")
    for item, value in table_i_rows:
        tex.append(f"{latex_escape(item)} & {latex_escape(value)} \\\\")
    tex.append("\\hline")
    tex.append("\\end{tabular}%")
    tex.append("}")
    tex.append("\\end{table}")
    tex.append("")
    tex.append("\\begin{table}[t]")
    tex.append("\\caption{Distribution of valid samples by the effective number of people. The multi-person distribution is computed over skeleton-valid frames, and the number of people per frame is determined by valid BODY-38 skeleton instances rather than directory-level subject tokens.}")
    tex.append("\\label{tab:multi_person_distribution}")
    tex.append("\\centering")
    tex.append("\\resizebox{\\columnwidth}{!}{%")
    tex.append("\\begin{tabular}{lrr}")
    tex.append("\\hline")
    tex.append("Number of people & Frames & Human instances \\\\")
    tex.append("\\hline")
    for row in mp_rows:
        tex.append(f"{latex_escape(row[0])} & {row[1]} & {row[2]} \\\\")
    tex.append("\\hline")
    tex.append("\\end{tabular}%")
    tex.append("}")
    tex.append("\\end{table}")
    tex.append("")
    tex.append("\\begin{table*}[t]")
    tex.append(
        "\\caption{Scene and router-position distribution. For all router configurations, the WiFi receiver is aligned with the ZED-X camera optical center and is treated as the camera coordinate origin. "
        "The transmitter position is represented by a manually measured offset relative to this receiver/camera origin. "
        "One recorded router configuration appears only in background/no-person sessions and is not counted as a human reconstruction configuration."
        + (
            f" In addition, {other_router_count} recorded configuration{'s' if other_router_count != 1 else ''} contain recorded sessions but no retained multimodal human samples under the current filtering rules."
            if other_router_count > 0
            else ""
        )
        + "}"
    )
    tex.append("\\label{tab:router_distribution}")
    tex.append("\\centering")
    tex.append("\\resizebox{\\textwidth}{!}{%")
    tex.append("\\begin{tabular}{llllrrl}")
    tex.append("\\hline")
    tex.append("Scene & Router position & Config ID & Sessions & Valid frames & Human instances & Type \\\\")
    tex.append("\\hline")
    for row in router_rows:
        tex.append(
            f"{latex_escape(row['scene'])} & {latex_escape(row['router_position'])} & {row['router_config_id']} & "
            f"{row['sessions']} & {row['valid_frames']} & {row['human_instances']} & {row['type']} \\\\"
        )
    tex.append("\\hline")
    tex.append("\\end{tabular}%")
    tex.append("}")
    tex.append("\\end{table*}")
    tex.append("")
    tex.append("\\begin{table}[t]")
    tex.append("\\caption{Paired physical-occlusion subset organized by occluder material and distance. Valid paired frames denote occluded--unoccluded pairs retained after quality control.}")
    tex.append("\\label{tab:physical_occlusion_dataset}")
    tex.append("\\centering")
    tex.append("\\resizebox{\\columnwidth}{!}{%")
    tex.append("\\begin{tabular}{lrrrrr}")
    tex.append("\\hline")
    tex.append("Occluder & Distance & Sessions & Raw paired frames & Valid paired frames & Discarded frames \\\\")
    tex.append("\\hline")
    for row in phys_rows:
        tex.append(
            f"{latex_escape(row[0])} & {latex_escape(row[1])} & {latex_escape(row[2])} & "
            f"{latex_escape(row[3])} & {latex_escape(row[4])} & {latex_escape(row[5])} \\\\"
        )
    tex.append("\\hline")
    tex.append("\\end{tabular}%")
    tex.append("}")
    tex.append("\\end{table}")
    tex.append("")
    tex.append("\\begin{table*}[t]")
    tex.append("\\caption{Evaluation protocols used in RF-Avatar experiments. Synthetic visual occlusion is not an independently collected dataset; it is generated online or during stress testing from clean RGB-D samples, while CSI remains unchanged.}")
    tex.append("\\label{tab:evaluation_protocols}")
    tex.append("\\centering")
    tex.append("\\resizebox{\\textwidth}{!}{%")
    tex.append("\\begin{tabular}{p{3.1cm}p{4.0cm}p{4.0cm}p{5.0cm}}")
    tex.append("\\hline")
    tex.append("Protocol & Training data & Testing data & Purpose \\\\")
    tex.append("\\hline")
    for row in eval_rows:
        tex.append(" & ".join(latex_escape(cell) for cell in row) + " \\\\")
    tex.append("\\hline")
    tex.append("\\end{tabular}%")
    tex.append("}")
    tex.append("\\end{table*}")
    tex.append("")
    tex.append("\\begin{table}[t]")
    tex.append("\\caption{Draft session-level train/validation/test split statistics derived from existing summary files. These totals are inconsistent with the dataset-level human-instance statistics and should not be used in the main paper until recomputed.}")
    tex.append("\\label{tab:split_stats}")
    tex.append("\\centering")
    tex.append("\\resizebox{\\columnwidth}{!}{%")
    tex.append("\\begin{tabular}{lrrrrr}")
    tex.append("\\hline")
    tex.append("Split & Sessions & RGB-D frames & Skeleton-valid frames & Human instances & Physical-occlusion pairs \\\\")
    tex.append("\\hline")
    for row in split_rows:
        tex.append(
            f"{latex_escape(row[0])} & {latex_escape(row[1])} & {latex_escape(row[2])} & "
            f"{latex_escape(row[3])} & {latex_escape(row[4])} & {latex_escape(row[5])} \\\\"
        )
    tex.append("\\hline")
    tex.append("\\end{tabular}%")
    tex.append("}")
    tex.append("\\end{table}")
    tex.append("")
    tex.append("\\begin{table}[t]")
    tex.append("\\caption{Appendix Table A: subject-level data distribution. In multi-person sessions, a recording session may be attributed to multiple participating subjects; therefore, subject-level totals are not expected to sum to the dataset-level frame count. Raw action categories are parsed from directory or metadata names and may include occlusion-related suffixes.}")
    tex.append("\\label{tab:subject_action_appendix}")
    tex.append("\\centering")
    tex.append("\\resizebox{\\columnwidth}{!}{%")
    tex.append("\\begin{tabular}{lrrrr}")
    tex.append("\\hline")
    tex.append("Subject & Action categories & Sessions & Valid frames & Human instances \\\\")
    tex.append("\\hline")
    for row in appendix_a_rows:
        tex.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\")
    tex.append("\\hline")
    tex.append("\\end{tabular}%")
    tex.append("}")
    tex.append("\\end{table}")
    tex.append("")
    tex.append("\\begin{table}[t]")
    tex.append("\\caption{Appendix Table B: quality-control statistics derived from the current dataset scan. These statistics are kept for appendix or internal reporting because several fields such as ghost skeleton removal and NaN/Inf checks are unavailable or not checked.}")
    tex.append("\\label{tab:quality_control_appendix}")
    tex.append("\\centering")
    tex.append("\\resizebox{\\columnwidth}{!}{%")
    tex.append("\\begin{tabular}{lr}")
    tex.append("\\hline")
    tex.append("Item & Count / Status \\\\")
    tex.append("\\hline")
    for row in appendix_b_rows:
        tex.append(f"{latex_escape(row[0])} & {latex_escape(row[1])} \\\\")
    tex.append("\\hline")
    tex.append("\\end{tabular}%")
    tex.append("}")
    tex.append("\\end{table}")
    write_text(output_dir / "paper_dataset_tables.tex", "\n".join(tex) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

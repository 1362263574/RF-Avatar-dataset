# -*- coding: utf-8 -*-
"""Analyze local RF-Avatar dataset statistics without modifying raw data.

The script is intentionally filesystem-first so it can work even when the
training codebase is unavailable. If a local ``dataset.py`` / ``RFAvatarDataset``
is found, that fact is recorded in the summary, but the statistics still fall
back to deterministic directory scanning.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal envs
    def tqdm(iterable=None, **_: Any):  # type: ignore
        return iterable


BODY38_JOINTS = 38
REGULAR_ROOT_NAME = "无遮挡"
PHYSICAL_ROOT_NAME = "遮挡"
SUMMARY_JSON_NAME = "dataset_summary.json"
SUMMARY_MD_NAME = "dataset_summary.md"
LATEX_TABLES_NAME = "latex_tables.md"
ADDITIONAL_REGULAR_ROOT_SUFFIX = "_row"


@dataclass
class FileCheck:
    exists: bool
    loadable: bool
    shape_ok: bool
    shape: Optional[Tuple[int, ...]] = None
    has_nan_or_inf: Optional[bool] = None
    message: str = ""


@dataclass
class CSIFileCheck:
    exists: bool
    loadable: bool
    keys_ok: bool
    shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    message: str = ""

    @property
    def valid(self) -> bool:
        return self.exists and self.loadable and self.keys_ok


@dataclass
class PoseFrameInfo:
    num_people: int
    incomplete: bool
    has_non_finite: bool
    person_ids: List[Any]


@dataclass
class RegularFrameSample:
    session_id: str
    scene_name: str
    router_position: str
    router_config_key: str
    subject_group: str
    subjects: List[str]
    expected_people: int
    action_name: str
    timestamp: str
    raw_candidate: bool
    rgb_exists: bool
    depth_check: FileCheck
    csi_check: CSIFileCheck
    pose_exists: bool
    pose_info: Optional[PoseFrameInfo]
    point_cloud_check: Optional[FileCheck]

    @property
    def rgbd_valid(self) -> bool:
        return self.rgb_exists and self.depth_check.exists and self.depth_check.loadable and self.depth_check.shape_ok

    @property
    def csi_valid(self) -> bool:
        return self.csi_check.valid

    @property
    def skeleton_valid(self) -> bool:
        return self.pose_exists and self.pose_info is not None

    @property
    def point_cloud_valid(self) -> bool:
        return self.point_cloud_check is not None and self.point_cloud_check.exists and self.point_cloud_check.loadable and self.point_cloud_check.shape_ok

    @property
    def valid_sample(self) -> bool:
        return self.rgbd_valid and self.csi_valid and self.skeleton_valid

    @property
    def valid_human_sample(self) -> bool:
        if not self.valid_sample or self.pose_info is None:
            return False
        if self.expected_people <= 0:
            return False
        return self.pose_info.num_people == self.expected_people and 1 <= self.expected_people <= 3


@dataclass
class RegularSession:
    session_id: str
    session_path: Path
    scene_router_dir: str
    scene_name: str
    router_position: str
    subject_group: str
    subject_tokens: List[str]
    action_name: str
    clip_meta: Dict[str, Any]
    rx_coords: Optional[List[float]]
    tx_coords: Optional[List[float]]
    tx_coord_semantics: str
    samples: List[RegularFrameSample]


@dataclass
class PhysicalPairEntry:
    session_id: str
    session_path: Path
    scene_name: str
    router_position: str
    router_config_key: str
    subject_group: str
    subjects: List[str]
    expected_people: int
    action_name: str
    occluder_type: str
    distance_cm: str
    frame_index: Optional[int]
    ts_occ: Optional[str]
    ts_unocc: Optional[str]
    rgb_occ_exists: bool
    depth_occ_check: FileCheck
    csi_occ_check: CSIFileCheck
    csi_unocc_check: CSIFileCheck
    pose_exists: bool
    pose_info: Optional[PoseFrameInfo]
    point_cloud_check: Optional[FileCheck]
    derived_from_perfect_list: bool

    @property
    def rgbd_valid(self) -> bool:
        return self.rgb_occ_exists and self.depth_occ_check.exists and self.depth_occ_check.loadable and self.depth_occ_check.shape_ok

    @property
    def csi_valid(self) -> bool:
        return self.csi_occ_check.valid and self.csi_unocc_check.valid

    @property
    def skeleton_valid(self) -> bool:
        return self.pose_exists and self.pose_info is not None

    @property
    def point_cloud_valid(self) -> bool:
        return self.point_cloud_check is not None and self.point_cloud_check.exists and self.point_cloud_check.loadable and self.point_cloud_check.shape_ok

    @property
    def valid_sample(self) -> bool:
        return self.rgbd_valid and self.csi_valid and self.skeleton_valid and self.point_cloud_valid

    @property
    def valid_human_sample(self) -> bool:
        if not self.valid_sample or self.pose_info is None:
            return False
        if self.expected_people <= 0:
            return False
        return self.pose_info.num_people == self.expected_people and 1 <= self.expected_people <= 3


@dataclass
class PhysicalPairSession:
    session_id: str
    session_path: Path
    scene_router_dir: str
    scene_name: str
    router_position: str
    subject_group: str
    subject_tokens: List[str]
    action_name: str
    occluder_type: str
    distance_cm: str
    occ_clip_meta: Dict[str, Any]
    unocc_clip_meta: Dict[str, Any]
    rx_coords: Optional[List[float]]
    tx_coords: Optional[List[float]]
    tx_coord_semantics: str
    estimated_raw_pairs: int
    perfect_entries: List[PhysicalPairEntry]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze local RF-Avatar dataset statistics.")
    parser.add_argument("--data-root", required=True, help="Path to the dataset root directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to store statistics outputs.")
    parser.add_argument(
        "--check-finite-arrays",
        action="store_true",
        help="Expensively scan npy/npz contents for NaN/Inf where applicable. Disabled by default.",
    )
    return parser.parse_args()


def canonical_path(path: Path) -> str:
    return path.resolve().as_posix()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_subject_tokens(name: str) -> List[str]:
    tokens = [token.upper() for token in re.findall(r"[KU]\d+", name, flags=re.IGNORECASE)]
    return sorted(set(tokens))


def parse_scene_router(scene_router_dir: str) -> Tuple[str, str]:
    if "_" not in scene_router_dir:
        return scene_router_dir.lower(), scene_router_dir
    prefix, suffix = scene_router_dir.split("_", 1)
    return prefix.lower(), suffix


def normalize_scene_name(meta_scene: Any, path_scene_name: str, *, is_physical: bool) -> str:
    path_scene = str(path_scene_name).lower().strip()
    meta = str(meta_scene or "").lower().strip().replace("\\", "")
    meta_compact = re.sub(r"[^a-z]", "", meta)
    if path_scene == "room":
        return "room"
    if is_physical:
        if meta_compact in {"newlab", "newlba", "newlan", "newab", "newlbab", "nealab", "nelab", "bewlab"}:
            return "newlab"
        return "newlab" if path_scene == "lab" else path_scene
    if meta_compact == "lab":
        return "lab"
    if meta_compact == "room":
        return "room"
    return path_scene


def coords_to_str(coords: Optional[List[float]], *, label: Optional[str] = None) -> str:
    if coords is None:
        return "unknown"
    payload = json.dumps(coords, ensure_ascii=False)
    if label:
        return f"{label}:{payload}"
    return payload


def normalize_occluder_type(name: str) -> str:
    lowered = name.lower()
    if "pop" in lowered:
        return "foam board"
    if "blackcloth" in lowered or "blockcloth" in lowered:
        return "black cloth"
    if "carton" in lowered or "cardboard" in lowered:
        return "cardboard board"
    return "unknown"


def extract_distance_cm(name: str) -> str:
    match = re.search(r"(\d+)\s*cm", name.lower())
    return match.group(1) if match else "unknown"


def is_background_subject_group(name: str) -> bool:
    lowered = name.lower()
    return lowered == "nothing" or "nothing" in lowered


def is_background_action(name: str) -> bool:
    lowered = name.lower()
    return lowered == "nothing" or lowered.endswith("_nothing") or "nothing" in lowered


def is_occlusion_free_action(name: str) -> bool:
    lowered = name.lower()
    return "occlusion" not in lowered and not is_background_action(lowered)


def normalize_metric_value(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def safe_rel_session_id(data_root: Path, session_path: Path) -> str:
    try:
        return session_path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return canonical_path(session_path)


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def probe_dataset_loader(workspace_root: Path) -> Dict[str, Any]:
    findings: Dict[str, Any] = {
        "found_dataset_py": False,
        "found_rfavatar_dataset_class": False,
        "dataset_py_paths": [],
        "notes": [],
    }
    dataset_files = list(workspace_root.rglob("dataset.py"))
    findings["dataset_py_paths"] = [canonical_path(path) for path in dataset_files]
    findings["found_dataset_py"] = bool(dataset_files)
    if not dataset_files:
        findings["notes"].append("No local dataset.py was found under the current workspace.")
        return findings

    for dataset_file in dataset_files:
        try:
            spec = importlib.util.spec_from_file_location("rfavatar_dataset_module", dataset_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "RFAvatarDataset"):
                findings["found_rfavatar_dataset_class"] = True
                findings["notes"].append(f"Detected RFAvatarDataset in {canonical_path(dataset_file)}.")
                break
        except Exception as exc:  # pragma: no cover - best effort probe
            findings["notes"].append(f"Failed to import {canonical_path(dataset_file)}: {exc}")
    if not findings["found_rfavatar_dataset_class"]:
        findings["notes"].append("No importable RFAvatarDataset class was detected; using filesystem-based scanning.")
    return findings


class DatasetAnalyzer:
    def __init__(self, data_root: Path, output_dir: Path, check_finite_arrays: bool = False) -> None:
        self.data_root = data_root.resolve()
        self.output_dir = output_dir.resolve()
        self.check_finite_arrays = check_finite_arrays
        self.regular_root = self.data_root / REGULAR_ROOT_NAME
        self.additional_regular_roots = self.discover_additional_regular_roots()
        self.physical_root = self.data_root / PHYSICAL_ROOT_NAME
        self.pose_cache: Dict[str, Dict[str, Any]] = {}
        self.depth_cache: Dict[str, FileCheck] = {}
        self.pc_cache: Dict[str, FileCheck] = {}
        self.csi_cache: Dict[str, CSIFileCheck] = {}
        self.dataset_loader_probe = probe_dataset_loader(self.data_root)
        self.perfect_train_entries: List[Dict[str, Any]] = []
        self.perfect_train_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.parsing_notes: List[str] = []
        self.quality_notes: List[str] = []
        self.manual_confirmations: List[str] = []
        self.inference_rules: List[str] = []

    def discover_additional_regular_roots(self) -> List[Path]:
        roots: List[Path] = []
        for child in sorted(self.data_root.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            if child.name == REGULAR_ROOT_NAME or child.name == PHYSICAL_ROOT_NAME:
                continue
            if child.name.endswith(ADDITIONAL_REGULAR_ROOT_SUFFIX):
                roots.append(child)
        return roots

    def run(self) -> None:
        ensure_dir(self.output_dir)
        self.load_perfect_train_list()
        regular_sessions = self.scan_regular_sessions()
        physical_sessions = self.scan_physical_sessions()
        summary = self.build_outputs(regular_sessions, physical_sessions)
        self.write_outputs(summary)

    def load_perfect_train_list(self) -> None:
        candidate = self.physical_root / "Lab_Pos2" / "perfect_train_list.json"
        if not candidate.exists():
            self.parsing_notes.append("No perfect_train_list.json was found under the physical occlusion subset.")
            return
        self.perfect_train_entries = json.loads(candidate.read_text(encoding="utf-8"))
        for entry in self.perfect_train_entries:
            localized_session = self.localize_external_path(entry.get("session", ""))
            if localized_session is None:
                continue
            session_id = safe_rel_session_id(self.data_root, localized_session)
            normalized = dict(entry)
            normalized["_localized_session_path"] = canonical_path(localized_session)
            self.perfect_train_map[session_id].append(normalized)
        self.parsing_notes.append(
            f"Loaded perfect_train_list.json with {len(self.perfect_train_entries)} paired entries for physical occlusion statistics."
        )

    def localize_external_path(self, raw_path: str) -> Optional[Path]:
        if not raw_path:
            return None
        raw_posix = PurePosixPath(raw_path)
        parts = list(raw_posix.parts)
        if not parts:
            return None
        if raw_path.startswith(str(self.data_root)):
            return Path(raw_path)
        start_idx = None
        for idx, token in enumerate(parts):
            if token.startswith("Lab_Pos") or token.startswith("Room_Pos"):
                start_idx = idx
                break
        if start_idx is None:
            return None
        tail = parts[start_idx:]
        preferred = self.data_root / PHYSICAL_ROOT_NAME / Path(*tail)
        if preferred.exists():
            return preferred
        alternate = self.data_root / REGULAR_ROOT_NAME / Path(*tail)
        if alternate.exists():
            return alternate
        for extra_root in self.additional_regular_roots:
            extra_candidate = extra_root / Path(*tail)
            if extra_candidate.exists():
                return extra_candidate
        fallback = self.data_root / Path(*tail)
        return fallback

    def load_pose_json(self, pose_path: Path) -> Dict[str, Any]:
        key = canonical_path(pose_path)
        if key not in self.pose_cache:
            if not pose_path.exists():
                self.pose_cache[key] = {}
            else:
                self.pose_cache[key] = read_json(pose_path)
        return self.pose_cache[key]

    def check_depth_file(self, path: Path) -> FileCheck:
        key = canonical_path(path)
        if key in self.depth_cache:
            return self.depth_cache[key]
        if not path.exists():
            check = FileCheck(False, False, False, message="missing depth file")
            self.depth_cache[key] = check
            return check
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = tuple(int(dim) for dim in arr.shape)
            shape_ok = arr.ndim == 2 and all(dim > 0 for dim in shape)
            has_nan_or_inf: Optional[bool] = None
            if self.check_finite_arrays:
                finite_mask = np.isfinite(np.asarray(arr))
                has_nan_or_inf = not bool(finite_mask.all())
            check = FileCheck(True, True, shape_ok, shape=shape, has_nan_or_inf=has_nan_or_inf)
        except Exception as exc:
            check = FileCheck(True, False, False, message=str(exc))
        self.depth_cache[key] = check
        return check

    def check_point_cloud_file(self, path: Path) -> FileCheck:
        key = canonical_path(path)
        if key in self.pc_cache:
            return self.pc_cache[key]
        if not path.exists():
            check = FileCheck(False, False, False, message="missing point cloud file")
            self.pc_cache[key] = check
            return check
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = tuple(int(dim) for dim in arr.shape)
            shape_ok = arr.ndim == 2 and len(shape) == 2 and shape[0] > 0 and shape[1] >= 3
            has_nan_or_inf: Optional[bool] = None
            if self.check_finite_arrays:
                finite_mask = np.isfinite(np.asarray(arr))
                has_nan_or_inf = not bool(finite_mask.all())
            check = FileCheck(True, True, shape_ok, shape=shape, has_nan_or_inf=has_nan_or_inf)
        except Exception as exc:
            check = FileCheck(True, False, False, message=str(exc))
        self.pc_cache[key] = check
        return check

    def check_csi_file(self, path: Path) -> CSIFileCheck:
        key = canonical_path(path)
        if key in self.csi_cache:
            return self.csi_cache[key]
        if not path.exists():
            check = CSIFileCheck(False, False, False, message="missing csi file")
            self.csi_cache[key] = check
            return check
        try:
            payload = np.load(path, allow_pickle=False)
            required_keys = {"amplitude", "phase"}
            keys_ok = required_keys.issubset(set(payload.files))
            shapes: Dict[str, Tuple[int, ...]] = {}
            for item in payload.files:
                shapes[item] = tuple(int(dim) for dim in payload[item].shape)
            check = CSIFileCheck(True, True, keys_ok, shapes=shapes)
        except Exception as exc:
            check = CSIFileCheck(True, False, False, message=str(exc))
        self.csi_cache[key] = check
        return check

    def analyze_pose_frame(self, people: Any) -> PoseFrameInfo:
        if not isinstance(people, list):
            return PoseFrameInfo(0, True, False, [])
        incomplete = False
        has_non_finite = False
        ids: List[Any] = []
        for person in people:
            if isinstance(person, dict):
                ids.append(person.get("id"))
                keypoints = person.get("keypoints_3d", [])
            else:
                ids.append(None)
                keypoints = []
            if not isinstance(keypoints, list) or len(keypoints) != BODY38_JOINTS:
                incomplete = True
                continue
            for keypoint in keypoints:
                if not isinstance(keypoint, list) or len(keypoint) != 3:
                    incomplete = True
                    continue
                for value in keypoint:
                    if isinstance(value, (int, float)) and not math.isfinite(value):
                        incomplete = True
                        has_non_finite = True
        return PoseFrameInfo(len(people), incomplete, has_non_finite, ids)

    def extract_router_coords(self, clip_meta: Dict[str, Any]) -> Tuple[Optional[List[float]], Optional[List[float]], str]:
        geometry = clip_meta.get("geometry_calibration", {}) if isinstance(clip_meta, dict) else {}
        rx_coords = geometry.get("rx_antenna_coord_mm")
        tx_coords = geometry.get("tx_antenna_offset_mm")
        semantics = "tx_antenna_offset_mm"
        if not isinstance(rx_coords, list):
            rx_coords = None
        if not isinstance(tx_coords, list):
            tx_coords = None
            semantics = "unknown"
        return rx_coords, tx_coords, semantics

    def scan_regular_sessions(self) -> List[RegularSession]:
        sessions: List[RegularSession] = []
        regular_roots = [root for root in [self.regular_root, *self.additional_regular_roots] if root.exists()]
        if not regular_roots:
            self.parsing_notes.append(
                f"No regular subset root was found under {canonical_path(self.data_root)}; expected {REGULAR_ROOT_NAME} and/or *_row directories."
            )
            return sessions

        for regular_root in regular_roots:
            scene_dirs = sorted([item for item in regular_root.iterdir() if item.is_dir()], key=lambda p: p.name)
            for scene_dir in tqdm(scene_dirs, desc=f"Scanning regular scenes ({regular_root.name})"):
                path_scene_name, router_position = parse_scene_router(scene_dir.name)
                subject_dirs = sorted([item for item in scene_dir.iterdir() if item.is_dir()], key=lambda p: p.name)
                for subject_dir in subject_dirs:
                    direct_session = (subject_dir / "clip_meta.json").exists() or (subject_dir / "visual").exists()
                    action_dirs = [subject_dir] if direct_session else sorted([item for item in subject_dir.iterdir() if item.is_dir()], key=lambda p: p.name)
                    for action_dir in action_dirs:
                        clip_meta_path = action_dir / "clip_meta.json"
                        visual_dir = action_dir / "visual"
                        pose_path = visual_dir / "pose_3d_gt.json"
                        csi_path = action_dir / "csi_clean.npz"
                        if not clip_meta_path.exists() and not visual_dir.exists():
                            continue
                        clip_meta = read_json(clip_meta_path) if clip_meta_path.exists() else {}
                        scene_name = normalize_scene_name(clip_meta.get("scene"), path_scene_name, is_physical=False)
                        rx_coords, tx_coords, tx_semantics = self.extract_router_coords(clip_meta)
                        subject_tokens = parse_subject_tokens(subject_dir.name)
                        action_name = action_dir.name if action_dir != subject_dir else str(clip_meta.get("action") or subject_dir.name)
                        session_id = safe_rel_session_id(self.data_root, action_dir)
                        samples = self.scan_regular_session_samples(
                            session_id=session_id,
                            session_path=action_dir,
                            scene_name=scene_name,
                            router_position=router_position,
                            rx_coords=rx_coords,
                            tx_coords=tx_coords,
                            tx_semantics=tx_semantics,
                            subject_group=subject_dir.name,
                            subject_tokens=subject_tokens,
                            action_name=action_name,
                            pose_path=pose_path,
                            visual_dir=visual_dir,
                            csi_path=csi_path,
                        )
                        sessions.append(
                            RegularSession(
                                session_id=session_id,
                                session_path=action_dir,
                                scene_router_dir=scene_dir.name,
                                scene_name=scene_name,
                                router_position=router_position,
                                subject_group=subject_dir.name,
                                subject_tokens=subject_tokens,
                                action_name=action_name,
                                clip_meta=clip_meta,
                                rx_coords=rx_coords,
                                tx_coords=tx_coords,
                                tx_coord_semantics=tx_semantics,
                                samples=samples,
                            )
                        )
        return sessions

    def scan_regular_session_samples(
        self,
        *,
        session_id: str,
        session_path: Path,
        scene_name: str,
        router_position: str,
        rx_coords: Optional[List[float]],
        tx_coords: Optional[List[float]],
        tx_semantics: str,
        subject_group: str,
        subject_tokens: List[str],
        action_name: str,
        pose_path: Path,
        visual_dir: Path,
        csi_path: Path,
    ) -> List[RegularFrameSample]:
        pose_payload = self.load_pose_json(pose_path) if pose_path.exists() else {}
        rgb_ts: Set[str] = set()
        depth_ts: Set[str] = set()
        pc_ts: Set[str] = set()
        if visual_dir.exists():
            for child in visual_dir.iterdir():
                if not child.is_file():
                    continue
                name = child.name
                if name.endswith("_rgb.png"):
                    rgb_ts.add(name[:-8])
                elif name.endswith("_depth.npy"):
                    depth_ts.add(name[:-10])
                elif name.endswith("_pc.npy"):
                    pc_ts.add(name[:-7])
        pose_ts = set(pose_payload.keys())
        candidate_ts = sorted(rgb_ts | depth_ts | pc_ts | pose_ts, key=self.safe_float_sort_key)
        router_config_key = self.make_router_config_key(scene_name, router_position, tx_coords, rx_coords, tx_semantics)
        csi_check = self.check_csi_file(csi_path)

        samples: List[RegularFrameSample] = []
        for timestamp in candidate_ts:
            rgb_path = visual_dir / f"{timestamp}_rgb.png"
            depth_path = visual_dir / f"{timestamp}_depth.npy"
            pc_path = visual_dir / f"{timestamp}_pc.npy"
            pose_exists = timestamp in pose_payload
            pose_info = self.analyze_pose_frame(pose_payload[timestamp]) if pose_exists else None
            point_cloud_check = self.check_point_cloud_file(pc_path) if timestamp in pc_ts else None
            samples.append(
                RegularFrameSample(
                    session_id=session_id,
                    scene_name=scene_name,
                    router_position=router_position,
                    router_config_key=router_config_key,
                    subject_group=subject_group,
                    subjects=subject_tokens,
                    expected_people=len(subject_tokens),
                    action_name=action_name,
                    timestamp=timestamp,
                    raw_candidate=True,
                    rgb_exists=rgb_path.exists(),
                    depth_check=self.check_depth_file(depth_path),
                    csi_check=csi_check,
                    pose_exists=pose_exists,
                    pose_info=pose_info,
                    point_cloud_check=point_cloud_check,
                )
            )
        return samples

    def safe_float_sort_key(self, value: str) -> Tuple[int, Any]:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    def make_router_config_key(
        self,
        scene_name: str,
        router_position: str,
        tx_coords: Optional[List[float]],
        rx_coords: Optional[List[float]],
        tx_semantics: str,
    ) -> str:
        tx_repr = coords_to_str(tx_coords, label=tx_semantics if tx_coords is not None else None)
        rx_repr = coords_to_str(rx_coords)
        return f"{scene_name}|{router_position}|{tx_repr}|{rx_repr}"

    def scan_physical_sessions(self) -> List[PhysicalPairSession]:
        sessions: List[PhysicalPairSession] = []
        if not self.physical_root.exists():
            self.parsing_notes.append(f"Physical occlusion subset root not found: {canonical_path(self.physical_root)}")
            return sessions

        scene_dirs = sorted([item for item in self.physical_root.iterdir() if item.is_dir()], key=lambda p: p.name)
        for scene_dir in tqdm(scene_dirs, desc="Scanning physical occlusion scenes"):
            path_scene_name, router_position = parse_scene_router(scene_dir.name)
            for subject_dir in sorted([item for item in scene_dir.iterdir() if item.is_dir()], key=lambda p: p.name):
                for occluder_dir in sorted([item for item in subject_dir.iterdir() if item.is_dir()], key=lambda p: p.name):
                    for distance_dir in sorted([item for item in occluder_dir.iterdir() if item.is_dir()], key=lambda p: p.name):
                        for action_dir in sorted([item for item in distance_dir.iterdir() if item.is_dir()], key=lambda p: p.name):
                            occ_dir = action_dir / "occluded"
                            unocc_dir = action_dir / "unoccluded"
                            if not occ_dir.exists() and not unocc_dir.exists():
                                continue
                            occ_clip_meta = read_json(occ_dir / "clip_meta.json") if (occ_dir / "clip_meta.json").exists() else {}
                            unocc_clip_meta = read_json(unocc_dir / "clip_meta.json") if (unocc_dir / "clip_meta.json").exists() else {}
                            scene_name = normalize_scene_name(
                                occ_clip_meta.get("scene") or unocc_clip_meta.get("scene"),
                                path_scene_name,
                                is_physical=True,
                            )
                            rx_coords, tx_coords, tx_semantics = self.extract_router_coords(occ_clip_meta or unocc_clip_meta)
                            session_id = safe_rel_session_id(self.data_root, action_dir)
                            subject_tokens = parse_subject_tokens(subject_dir.name)
                            perfect_entries = self.scan_physical_pair_entries(
                                session_id=session_id,
                                session_path=action_dir,
                                scene_name=scene_name,
                                router_position=router_position,
                                subject_group=subject_dir.name,
                                subject_tokens=subject_tokens,
                                action_name=action_dir.name,
                                occ_dir=occ_dir,
                                unocc_dir=unocc_dir,
                                occluder_type=normalize_occluder_type(f"{occluder_dir.name}/{distance_dir.name}"),
                                distance_cm=extract_distance_cm(distance_dir.name),
                                rx_coords=rx_coords,
                                tx_coords=tx_coords,
                                tx_semantics=tx_semantics,
                            )
                            estimated_raw_pairs = self.estimate_physical_raw_pairs(occ_dir, unocc_dir)
                            sessions.append(
                                PhysicalPairSession(
                                    session_id=session_id,
                                    session_path=action_dir,
                                    scene_router_dir=scene_dir.name,
                                    scene_name=scene_name,
                                    router_position=router_position,
                                    subject_group=subject_dir.name,
                                    subject_tokens=subject_tokens,
                                    action_name=action_dir.name,
                                    occluder_type=normalize_occluder_type(f"{occluder_dir.name}/{distance_dir.name}"),
                                    distance_cm=extract_distance_cm(distance_dir.name),
                                    occ_clip_meta=occ_clip_meta,
                                    unocc_clip_meta=unocc_clip_meta,
                                    rx_coords=rx_coords,
                                    tx_coords=tx_coords,
                                    tx_coord_semantics=tx_semantics,
                                    estimated_raw_pairs=estimated_raw_pairs,
                                    perfect_entries=perfect_entries,
                                )
                            )
        return sessions

    def estimate_physical_raw_pairs(self, occ_dir: Path, unocc_dir: Path) -> int:
        occ_visual = occ_dir / "visual"
        unocc_visual = unocc_dir / "visual"
        occ_rgb = len(list(occ_visual.glob("*_rgb.png"))) if occ_visual.exists() else 0
        occ_depth = len(list(occ_visual.glob("*_depth.npy"))) if occ_visual.exists() else 0
        unocc_pose_count = 0
        pose_path = unocc_visual / "pose_3d_gt.json"
        if pose_path.exists():
            try:
                unocc_pose_count = len(self.load_pose_json(pose_path))
            except Exception:
                unocc_pose_count = 0
        counts = [count for count in (occ_rgb, occ_depth, unocc_pose_count) if count > 0]
        if not counts:
            return 0
        return min(counts)

    def scan_physical_pair_entries(
        self,
        *,
        session_id: str,
        session_path: Path,
        scene_name: str,
        router_position: str,
        subject_group: str,
        subject_tokens: List[str],
        action_name: str,
        occ_dir: Path,
        unocc_dir: Path,
        occluder_type: str,
        distance_cm: str,
        rx_coords: Optional[List[float]],
        tx_coords: Optional[List[float]],
        tx_semantics: str,
    ) -> List[PhysicalPairEntry]:
        router_config_key = self.make_router_config_key(scene_name, router_position, tx_coords, rx_coords, tx_semantics)
        perfect_entries = self.perfect_train_map.get(session_id, [])
        pose_path = unocc_dir / "visual" / "pose_3d_gt.json"
        pose_payload = self.load_pose_json(pose_path) if pose_path.exists() else {}
        if not perfect_entries:
            return []

        entries: List[PhysicalPairEntry] = []
        for entry in perfect_entries:
            rgb_occ = self.localize_external_path(entry.get("rgb_occ", "")) or (occ_dir / "visual" / f"{entry.get('ts_occ', '')}_rgb.png")
            depth_occ = self.localize_external_path(entry.get("depth_occ", "")) or (occ_dir / "visual" / f"{entry.get('ts_occ', '')}_depth.npy")
            csi_occ = self.localize_external_path(entry.get("csi_occ", "")) or (occ_dir / "csi_clean.npz")
            csi_unocc = self.localize_external_path(entry.get("csi_unocc", "")) or (unocc_dir / "csi_clean.npz")
            pc_gt = self.localize_external_path(entry.get("pc_gt", "")) or (unocc_dir / "visual" / f"{entry.get('ts_unocc', '')}_pc.npy")
            pose_gt_json = self.localize_external_path(entry.get("pose_gt_json", "")) or pose_path
            pose_exists = False
            pose_info: Optional[PoseFrameInfo] = None
            ts_unocc = str(entry.get("ts_unocc", ""))
            if pose_gt_json.exists():
                pose_payload = self.load_pose_json(pose_gt_json)
                if ts_unocc in pose_payload:
                    pose_exists = True
                    pose_info = self.analyze_pose_frame(pose_payload[ts_unocc])
            entries.append(
                PhysicalPairEntry(
                    session_id=session_id,
                    session_path=session_path,
                    scene_name=scene_name,
                    router_position=router_position,
                    router_config_key=router_config_key,
                    subject_group=subject_group,
                    subjects=subject_tokens,
                    expected_people=len(subject_tokens),
                    action_name=action_name,
                    occluder_type=occluder_type,
                    distance_cm=distance_cm,
                    frame_index=entry.get("frame_index"),
                    ts_occ=str(entry.get("ts_occ")) if entry.get("ts_occ") is not None else None,
                    ts_unocc=ts_unocc if ts_unocc else None,
                    rgb_occ_exists=rgb_occ.exists(),
                    depth_occ_check=self.check_depth_file(depth_occ),
                    csi_occ_check=self.check_csi_file(csi_occ),
                    csi_unocc_check=self.check_csi_file(csi_unocc),
                    pose_exists=pose_exists,
                    pose_info=pose_info,
                    point_cloud_check=self.check_point_cloud_file(pc_gt),
                    derived_from_perfect_list=True,
                )
            )
        return entries

    def build_outputs(
        self,
        regular_sessions: List[RegularSession],
        physical_sessions: List[PhysicalPairSession],
    ) -> Dict[str, Any]:
        subject_action_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
        action_rows: Dict[str, Dict[str, Any]] = {}
        people_distribution: Dict[int, Dict[str, int]] = {1: {"frames": 0, "human_instances": 0}, 2: {"frames": 0, "human_instances": 0}, 3: {"frames": 0, "human_instances": 0}}
        scene_router_rows: Dict[str, Dict[str, Any]] = {}
        scene_router_sessions: Dict[str, Set[str]] = defaultdict(set)
        unique_subjects: Set[str] = set()
        unique_subject_groups: Set[str] = set()
        unique_actions: Set[str] = set()
        unique_scenes: Set[str] = set()
        unique_router_configs: Set[str] = set()
        unique_occlusion_free_subjects: Set[str] = set()
        unique_occlusion_free_actions: Set[str] = set()
        unique_occlusion_free_scenes: Set[str] = set()
        unique_occlusion_free_router_configs: Set[str] = set()
        unique_occlusion_free_sessions: Set[str] = set()
        expected_people_mismatch_frames = 0
        background_sessions = 0

        overall = Counter()
        quality = {
            "total_raw_samples": 0,
            "valid_samples_after_filtering": 0,
            "missing_rgbd": 0,
            "invalid_depth": 0,
            "invalid_csi": 0,
            "missing_skeleton": 0,
            "incomplete_skeleton": 0,
            "ghost_skeleton_removed": "unknown",
            "invalid_point_cloud": 0,
            "nan_inf_samples": "unknown",
            "shape_error_samples": 0,
        }

        for session in regular_sessions:
            unique_actions.add(session.action_name)
            unique_scenes.add(session.scene_name)
            unique_router_configs.add(
                self.make_router_config_key(session.scene_name, session.router_position, session.tx_coords, session.rx_coords, session.tx_coord_semantics)
            )
            unique_subject_groups.add(session.subject_group)
            for subject in session.subject_tokens:
                unique_subjects.add(subject)
            is_background_session = is_background_subject_group(session.subject_group) or is_background_action(session.action_name)
            if is_background_session:
                background_sessions += 1

            scene_router_key = self.make_router_config_key(session.scene_name, session.router_position, session.tx_coords, session.rx_coords, session.tx_coord_semantics)
            if scene_router_key not in scene_router_rows:
                scene_router_rows[scene_router_key] = {
                    "scene_name": session.scene_name,
                    "router_position": session.router_position,
                    "tx_coords": coords_to_str(session.tx_coords, label=session.tx_coord_semantics if session.tx_coords is not None else None),
                    "rx_coords": coords_to_str(session.rx_coords),
                    "sessions": 0,
                    "background_sessions": 0,
                    "valid_frames": 0,
                    "human_instances": 0,
                }
            scene_router_sessions[scene_router_key].add(session.session_id)
            if is_background_session:
                scene_router_rows[scene_router_key]["background_sessions"] += 1
            if session.action_name not in action_rows:
                action_rows[session.action_name] = {
                    "action_name": session.action_name,
                    "sessions": set(),
                    "valid_frames": 0,
                    "human_instances": 0,
                    "subjects": set(),
                    "scenes": set(),
                    "router_configurations": set(),
                    "subset_types": set(),
                }
            action_rows[session.action_name]["sessions"].add(session.session_id)
            action_rows[session.action_name]["subjects"].update(session.subject_tokens)
            action_rows[session.action_name]["scenes"].add(session.scene_name)
            action_rows[session.action_name]["router_configurations"].add(scene_router_key)
            action_rows[session.action_name]["subset_types"].add("regular")

            for sample in session.samples:
                quality["total_raw_samples"] += 1
                overall["valid_rgbd_frames"] += int(sample.rgbd_valid)
                overall["valid_csi_samples"] += int(sample.rgbd_valid and sample.csi_valid)
                overall["valid_skeleton_annotations"] += int(sample.valid_human_sample)
                overall["valid_point_cloud_pseudo_labels"] += int(sample.valid_human_sample and sample.point_cloud_valid)

                if not sample.rgb_exists or not sample.depth_check.exists:
                    quality["missing_rgbd"] += 1
                if sample.depth_check.exists and not sample.depth_check.loadable:
                    quality["invalid_depth"] += 1
                if sample.depth_check.exists and sample.depth_check.loadable and not sample.depth_check.shape_ok:
                    quality["shape_error_samples"] += 1
                if sample.depth_check.exists and sample.depth_check.loadable and sample.depth_check.has_nan_or_inf:
                    quality["nan_inf_samples"] = quality.get("nan_inf_samples", 0)
                    if isinstance(quality["nan_inf_samples"], int):
                        quality["nan_inf_samples"] += 1
                if not sample.csi_check.valid:
                    quality["invalid_csi"] += 1
                if not sample.pose_exists and not is_background_subject_group(sample.subject_group):
                    quality["missing_skeleton"] += 1
                if sample.pose_info is not None and sample.pose_info.incomplete:
                    quality["incomplete_skeleton"] += 1
                if sample.point_cloud_check is not None:
                    if not sample.point_cloud_check.loadable or not sample.point_cloud_check.shape_ok:
                        quality["invalid_point_cloud"] += 1
                    if sample.point_cloud_check.has_nan_or_inf:
                        quality["nan_inf_samples"] = quality.get("nan_inf_samples", 0)
                        if isinstance(quality["nan_inf_samples"], int):
                            quality["nan_inf_samples"] += 1

                if not sample.valid_sample:
                    continue
                quality["valid_samples_after_filtering"] += 1
                num_people = sample.pose_info.num_people if sample.pose_info is not None else 0
                if sample.subjects and num_people != len(sample.subjects):
                    expected_people_mismatch_frames += 1
                if not sample.valid_human_sample:
                    continue

                overall["human_instances"] += num_people
                overall["max_people_per_frame"] = max(overall.get("max_people_per_frame", 0), num_people)

                if 1 <= num_people <= 3:
                    people_distribution[num_people]["frames"] += 1
                    people_distribution[num_people]["human_instances"] += num_people

                scene_router_rows[scene_router_key]["valid_frames"] += 1
                scene_router_rows[scene_router_key]["human_instances"] += num_people
                action_rows[sample.action_name]["valid_frames"] += 1
                action_rows[sample.action_name]["human_instances"] += num_people

                if sample.subjects:
                    for subject in sample.subjects:
                        row_key = (subject, sample.action_name)
                        if row_key not in subject_action_rows:
                            subject_action_rows[row_key] = {
                                "subject_id": subject,
                                "action_name": sample.action_name,
                                "sessions": set(),
                                "valid_frames": 0,
                                "human_instances": 0,
                            }
                        subject_action_rows[row_key]["sessions"].add(sample.session_id)
                        subject_action_rows[row_key]["valid_frames"] += 1
                        subject_action_rows[row_key]["human_instances"] += 1

                if sample.subjects and is_occlusion_free_action(sample.action_name):
                    unique_occlusion_free_sessions.add(sample.session_id)
                    unique_occlusion_free_actions.add(sample.action_name)
                    unique_occlusion_free_scenes.add(sample.scene_name)
                    unique_occlusion_free_router_configs.add(sample.router_config_key)
                    for subject in sample.subjects:
                        unique_occlusion_free_subjects.add(subject)
                    overall["occlusion_free_valid_frames"] += 1
                    overall["occlusion_free_human_instances"] += num_people

        for scene_router_key, session_ids in scene_router_sessions.items():
            scene_router_rows[scene_router_key]["sessions"] = len(session_ids)

        physical_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
        physical_router_sessions: Dict[str, Set[str]] = defaultdict(set)
        physical_train_sessions: Set[str] = set()
        physical_train_frames = 0
        physical_train_instances = 0

        for session in physical_sessions:
            unique_actions.add(session.action_name)
            unique_scenes.add(session.scene_name)
            unique_router_configs.add(
                self.make_router_config_key(session.scene_name, session.router_position, session.tx_coords, session.rx_coords, session.tx_coord_semantics)
            )
            unique_subject_groups.add(session.subject_group)
            for subject in session.subject_tokens:
                unique_subjects.add(subject)

            scene_router_key = self.make_router_config_key(session.scene_name, session.router_position, session.tx_coords, session.rx_coords, session.tx_coord_semantics)
            if scene_router_key not in scene_router_rows:
                scene_router_rows[scene_router_key] = {
                    "scene_name": session.scene_name,
                    "router_position": session.router_position,
                    "tx_coords": coords_to_str(session.tx_coords, label=session.tx_coord_semantics if session.tx_coords is not None else None),
                    "rx_coords": coords_to_str(session.rx_coords),
                    "sessions": 0,
                    "background_sessions": 0,
                    "valid_frames": 0,
                    "human_instances": 0,
                }
            physical_router_sessions[scene_router_key].add(session.session_id)
            if session.action_name not in action_rows:
                action_rows[session.action_name] = {
                    "action_name": session.action_name,
                    "sessions": set(),
                    "valid_frames": 0,
                    "human_instances": 0,
                    "subjects": set(),
                    "scenes": set(),
                    "router_configurations": set(),
                    "subset_types": set(),
                }
            action_rows[session.action_name]["sessions"].add(session.session_id)
            action_rows[session.action_name]["subjects"].update(session.subject_tokens)
            action_rows[session.action_name]["scenes"].add(session.scene_name)
            action_rows[session.action_name]["router_configurations"].add(scene_router_key)
            action_rows[session.action_name]["subset_types"].add("physical_occlusion")

            pair_key = (session.occluder_type, session.distance_cm)
            if pair_key not in physical_rows:
                physical_rows[pair_key] = {
                    "occluder_type": session.occluder_type,
                    "distance_cm": session.distance_cm,
                    "sessions": set(),
                    "raw_paired_frames": 0,
                    "valid_paired_frames": 0,
                    "discarded_frames": 0,
                }
            physical_rows[pair_key]["sessions"].add(session.session_id)
            physical_rows[pair_key]["raw_paired_frames"] += session.estimated_raw_pairs

            for entry in session.perfect_entries:
                quality["total_raw_samples"] += 1
                overall["valid_rgbd_frames"] += int(entry.rgbd_valid)
                overall["valid_csi_samples"] += int(entry.rgbd_valid and entry.csi_valid)
                overall["valid_skeleton_annotations"] += int(entry.valid_human_sample)
                overall["valid_point_cloud_pseudo_labels"] += int(entry.valid_human_sample and entry.point_cloud_valid)

                if not entry.rgb_occ_exists or not entry.depth_occ_check.exists:
                    quality["missing_rgbd"] += 1
                if entry.depth_occ_check.exists and not entry.depth_occ_check.loadable:
                    quality["invalid_depth"] += 1
                if entry.depth_occ_check.exists and entry.depth_occ_check.loadable and not entry.depth_occ_check.shape_ok:
                    quality["shape_error_samples"] += 1
                if not entry.csi_occ_check.valid or not entry.csi_unocc_check.valid:
                    quality["invalid_csi"] += 1
                if not entry.pose_exists:
                    quality["missing_skeleton"] += 1
                if entry.pose_info is not None and entry.pose_info.incomplete:
                    quality["incomplete_skeleton"] += 1
                if entry.point_cloud_check is not None and (not entry.point_cloud_check.loadable or not entry.point_cloud_check.shape_ok):
                    quality["invalid_point_cloud"] += 1

                if not entry.valid_sample:
                    continue
                quality["valid_samples_after_filtering"] += 1
                num_people = entry.pose_info.num_people if entry.pose_info is not None else 0
                if entry.subjects and num_people != len(entry.subjects):
                    expected_people_mismatch_frames += 1
                if not entry.valid_human_sample:
                    continue

                overall["human_instances"] += num_people
                overall["max_people_per_frame"] = max(overall.get("max_people_per_frame", 0), num_people)
                if 1 <= num_people <= 3:
                    people_distribution[num_people]["frames"] += 1
                    people_distribution[num_people]["human_instances"] += num_people
                scene_router_rows[scene_router_key]["valid_frames"] += 1
                scene_router_rows[scene_router_key]["human_instances"] += num_people
                action_rows[entry.action_name]["valid_frames"] += 1
                action_rows[entry.action_name]["human_instances"] += num_people
                physical_rows[pair_key]["valid_paired_frames"] += 1
                physical_train_sessions.add(entry.session_id)
                physical_train_frames += 1
                physical_train_instances += num_people

                for subject in entry.subjects:
                    row_key = (subject, entry.action_name)
                    if row_key not in subject_action_rows:
                        subject_action_rows[row_key] = {
                            "subject_id": subject,
                            "action_name": entry.action_name,
                            "sessions": set(),
                            "valid_frames": 0,
                            "human_instances": 0,
                        }
                    subject_action_rows[row_key]["sessions"].add(entry.session_id)
                    subject_action_rows[row_key]["valid_frames"] += 1
                    subject_action_rows[row_key]["human_instances"] += 1

        for scene_router_key, session_ids in physical_router_sessions.items():
            scene_router_rows[scene_router_key]["sessions"] += len(session_ids)

        for row in physical_rows.values():
            row["discarded_frames"] = max(0, row["raw_paired_frames"] - row["valid_paired_frames"])

        if not self.check_finite_arrays:
            quality["nan_inf_samples"] = "unknown"
            self.quality_notes.append(
                "NaN / Inf sample counting was left as unknown because --check-finite-arrays was not enabled and depth maps routinely contain NaNs as valid missing-depth markers."
            )
        if expected_people_mismatch_frames > 0:
            self.manual_confirmations.append(
                f"{expected_people_mismatch_frames} valid frames had a mismatch between subject-count parsed from the directory name and people-count parsed from pose_3d_gt.json."
            )

        number_of_router_deployment_configurations = len(unique_router_configs)
        number_of_recording_sessions = len(regular_sessions) + len(physical_sessions)

        subject_action_csv_rows = []
        for (_, _), row in sorted(subject_action_rows.items(), key=lambda item: (item[0][0], item[0][1])):
            subject_action_csv_rows.append(
                {
                    "subject_id": row["subject_id"],
                    "action_name": row["action_name"],
                    "sessions": len(row["sessions"]),
                    "valid_frames": row["valid_frames"],
                    "human_instances": row["human_instances"],
                }
            )
        action_csv_rows = []
        for action_name, row in sorted(action_rows.items(), key=lambda item: item[0]):
            action_csv_rows.append(
                {
                    "action_name": action_name,
                    "sessions": len(row["sessions"]),
                    "valid_frames": row["valid_frames"],
                    "human_instances": row["human_instances"],
                    "subjects": len(row["subjects"]),
                    "scenes": len(row["scenes"]),
                    "router_configurations": len(row["router_configurations"]),
                    "subset_types": ",".join(sorted(row["subset_types"])),
                }
            )

        people_distribution_rows = [
            {
                "num_people": num_people,
                "frames": stats["frames"],
                "human_instances": stats["human_instances"],
            }
            for num_people, stats in sorted(people_distribution.items())
        ]

        scene_router_csv_rows = sorted(scene_router_rows.values(), key=lambda row: (row["scene_name"], row["router_position"], row["tx_coords"], row["rx_coords"]))
        zero_human_router_rows = [row for row in scene_router_csv_rows if row["valid_frames"] == 0]

        occlusion_free_rows = [
            {"metric": "occlusion-free sessions", "value": len(unique_occlusion_free_sessions)},
            {"metric": "occlusion-free valid frames", "value": overall.get("occlusion_free_valid_frames", 0)},
            {"metric": "occlusion-free human instances", "value": overall.get("occlusion_free_human_instances", 0)},
            {"metric": "subjects", "value": len(unique_occlusion_free_subjects)},
            {"metric": "actions", "value": len(unique_occlusion_free_actions)},
            {"metric": "scenes", "value": len(unique_occlusion_free_scenes)},
            {"metric": "router configurations", "value": len(unique_occlusion_free_router_configs)},
        ]

        physical_csv_rows = sorted(
            (
                {
                    "occluder_type": row["occluder_type"],
                    "distance_cm": row["distance_cm"],
                    "sessions": len(row["sessions"]),
                    "raw_paired_frames": row["raw_paired_frames"],
                    "valid_paired_frames": row["valid_paired_frames"],
                    "discarded_frames": row["discarded_frames"],
                }
                for row in physical_rows.values()
            ),
            key=lambda row: (row["occluder_type"], row["distance_cm"]),
        )

        quality_rows = [
            {"metric": "total raw samples", "value": quality["total_raw_samples"], "notes": "Regular raw frame candidates plus inspected paired samples from perfect_train_list.json."},
            {"metric": "valid samples after filtering", "value": quality["valid_samples_after_filtering"], "notes": "Samples with valid RGB-D, CSI, and skeleton; physical pairs additionally require point cloud pseudo-labels."},
            {"metric": "missing RGB-D", "value": quality["missing_rgbd"], "notes": ""},
            {"metric": "invalid depth", "value": quality["invalid_depth"], "notes": "Counts missing/unloadable depth files only; NaN depth values are not automatically treated as invalid."},
            {"metric": "invalid CSI", "value": quality["invalid_csi"], "notes": "Counts missing/unloadable CSI or missing amplitude/phase keys."},
            {"metric": "missing skeleton", "value": quality["missing_skeleton"], "notes": ""},
            {"metric": "incomplete skeleton", "value": quality["incomplete_skeleton"], "notes": f"Frame-level count where BODY-38 keypoints were incomplete or contained non-finite values."},
            {"metric": "ghost skeleton removed", "value": quality["ghost_skeleton_removed"], "notes": "Cannot be recovered reliably from post-filtered files alone."},
            {"metric": "invalid point cloud", "value": quality["invalid_point_cloud"], "notes": ""},
            {"metric": "NaN / Inf samples", "value": quality["nan_inf_samples"], "notes": "Unknown unless --check-finite-arrays is enabled; depth NaNs often represent valid missing-depth pixels."},
            {"metric": "shape error samples", "value": quality["shape_error_samples"], "notes": "Counts malformed array shapes among inspected depth / point cloud files."},
        ]

        split_suggestion = self.generate_split_suggestion(regular_sessions, physical_sessions)
        split_stats_rows = split_suggestion["split_rows"]
        if self.perfect_train_entries:
            split_stats_rows.append(
                {
                    "split": "physical_train_existing",
                    "sessions": len(physical_train_sessions),
                    "frames": physical_train_frames,
                    "human_instances": physical_train_instances,
                    "source": "perfect_train_list.json",
                    "notes": "Existing paired physical-occlusion training list; no separate val/test list was found.",
                }
            )

        summary_counts = {
            "number_of_subjects": len(unique_subjects),
            "number_of_subject_groups": len(unique_subject_groups),
            "number_of_action_categories": len(unique_actions),
            "number_of_indoor_scenes": len(unique_scenes),
            "number_of_router_deployment_configurations": number_of_router_deployment_configurations,
            "number_of_recording_sessions": number_of_recording_sessions,
            "number_of_valid_rgbd_frames": overall.get("valid_rgbd_frames", 0),
            "number_of_valid_csi_samples": overall.get("valid_csi_samples", 0),
            "number_of_valid_body38_skeleton_annotations": overall.get("valid_skeleton_annotations", 0),
            "number_of_valid_human_point_cloud_pseudo_labels": overall.get("valid_point_cloud_pseudo_labels", 0),
            "number_of_human_instances": overall.get("human_instances", 0),
            "maximum_number_of_people_per_frame": overall.get("max_people_per_frame", 0),
            "background_sessions_detected": background_sessions,
        }

        self.inference_rules.extend(
            [
                "Subject IDs are parsed from directory tokens matching K<number> or U<number>; e.g., K1K3 or U1U2 expands to individual subjects, and each token is treated as a distinct person identity.",
                "subject_action_stats.csv attributes each valid frame in a multi-person session to every subject token parsed from that session directory.",
                "action_stats.csv aggregates each action over all sessions and reports valid frame count, human instance count, covered subjects, scenes, and router configurations.",
                "router_position is parsed from the scene-router directory suffix (e.g., Room_Pos1_L20 -> Pos1_L20).",
                "scene_name is normalized from clip_meta.json plus the scene-router directory prefix, so obvious typos such as romm/newlba are folded back to room/newlab.",
                "Regular-subset scanning includes the legacy 无遮挡 root plus any additional top-level *_row directories that match the same session layout.",
                "The WiFi receiver is co-located with the ZED-X camera optical center and is therefore treated as the camera coordinate origin; the transmitter position is represented by geometry_calibration.tx_antenna_offset_mm relative to that origin.",
                "Physical-occlusion valid paired frames prefer perfect_train_list.json; discarded paired frames are estimated as filesystem raw pairs minus valid perfect-list pairs.",
            ]
        )

        if self.dataset_loader_probe["found_rfavatar_dataset_class"]:
            self.parsing_notes.append("An RFAvatarDataset class was detected, but this script still used filesystem scanning to avoid depending on external project code during local dataset-only analysis.")
        else:
            self.parsing_notes.append("No usable RFAvatarDataset loader was found in the current workspace; all statistics are filesystem-derived.")
        if zero_human_router_rows:
            self.parsing_notes.append(
                f"{len(zero_human_router_rows)} router configuration(s) were observed only in zero-valid-human sessions; keep this in mind if the paper should report human-data-only router deployments."
            )

        self.manual_confirmations.extend(
            [
                "Confirm whether background 'nothing' sessions should be reported in the paper as auxiliary no-person negatives or excluded entirely from Experimental Setup tables.",
                "Confirm whether action names should follow directory names (e.g., sit_occlusion_lower) or clip_meta.json values when the two differ slightly.",
            ]
        )

        summary = {
            "data_root": canonical_path(self.data_root),
            "output_dir": canonical_path(self.output_dir),
            "dataset_loader_probe": self.dataset_loader_probe,
            "counts": summary_counts,
            "subject_action_stats": subject_action_csv_rows,
            "action_stats": action_csv_rows,
            "people_distribution": people_distribution_rows,
            "scene_router_stats": scene_router_csv_rows,
            "occlusion_free_stats": occlusion_free_rows,
            "physical_occlusion_stats": physical_csv_rows,
            "quality_control_stats": quality_rows,
            "split_stats": split_stats_rows,
            "split_suggestion": split_suggestion["split_json"],
            "parsing_notes": self.parsing_notes,
            "quality_notes": self.quality_notes,
            "manual_confirmations": self.manual_confirmations,
            "inference_rules": self.inference_rules,
        }
        return summary

    def generate_split_suggestion(
        self,
        regular_sessions: List[RegularSession],
        physical_sessions: List[PhysicalPairSession],
    ) -> Dict[str, Any]:
        session_records: List[Dict[str, Any]] = []
        for session in regular_sessions:
            valid_frames = sum(1 for sample in session.samples if sample.valid_sample and (sample.pose_info is None or sample.pose_info.num_people > 0))
            human_instances = sum(sample.pose_info.num_people for sample in session.samples if sample.valid_sample and sample.pose_info is not None)
            if valid_frames == 0:
                continue
            session_records.append(
                {
                    "session_id": session.session_id,
                    "kind": "regular",
                    "scene_name": session.scene_name,
                    "router_position": session.router_position,
                    "action_name": session.action_name,
                    "subjects": session.subject_tokens,
                    "num_people": len(session.subject_tokens),
                    "valid_frames": valid_frames,
                    "human_instances": human_instances,
                }
            )
        for session in physical_sessions:
            valid_frames = sum(1 for entry in session.perfect_entries if entry.valid_sample)
            human_instances = sum(entry.pose_info.num_people for entry in session.perfect_entries if entry.valid_sample and entry.pose_info is not None)
            if valid_frames == 0:
                continue
            session_records.append(
                {
                    "session_id": session.session_id,
                    "kind": "physical_occlusion",
                    "scene_name": session.scene_name,
                    "router_position": session.router_position,
                    "action_name": session.action_name,
                    "subjects": session.subject_tokens,
                    "num_people": len(session.subject_tokens),
                    "valid_frames": valid_frames,
                    "human_instances": human_instances,
                }
            )

        groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
        for record in session_records:
            groups[(record["kind"], record["scene_name"], record["num_people"])].append(record)

        split_assignments = {"train": [], "val": [], "test": []}
        target_order = ["train", "val", "test"]
        for group_key, records in sorted(groups.items(), key=lambda item: item[0]):
            records = sorted(records, key=lambda record: record["session_id"])
            n = len(records)
            n_train = max(1, round(n * 0.7))
            n_val = round(n * 0.15)
            n_test = n - n_train - n_val
            if n >= 3 and n_val == 0:
                n_val = 1
            if n >= 4 and n_test == 0:
                n_test = 1
            while n_train + n_val + n_test > n:
                if n_train >= max(n_val, n_test) and n_train > 1:
                    n_train -= 1
                elif n_val > 0:
                    n_val -= 1
                else:
                    n_test -= 1
            while n_train + n_val + n_test < n:
                n_train += 1
            quotas = {"train": n_train, "val": n_val, "test": n_test}
            index = 0
            for split in target_order:
                for _ in range(quotas[split]):
                    if index >= n:
                        break
                    split_assignments[split].append(records[index])
                    index += 1

        split_rows = []
        split_json: Dict[str, Any] = {
            "strategy": "session-level split suggestion to avoid frame-level leakage",
            "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "notes": [
                "Sessions are assigned as indivisible units so adjacent frames from the same clip do not leak across splits.",
                "The heuristic stratifies by subset kind, scene, and parsed people-count before applying approximate 70/15/15 allocation.",
            ],
            "splits": {},
        }
        for split_name, records in split_assignments.items():
            records = sorted(records, key=lambda record: record["session_id"])
            split_rows.append(
                {
                    "split": split_name,
                    "sessions": len(records),
                    "frames": sum(record["valid_frames"] for record in records),
                    "human_instances": sum(record["human_instances"] for record in records),
                    "source": "suggested_session_level",
                    "notes": "Suggested because no complete train/val/test list was found for the full dataset.",
                }
            )
            split_json["splits"][split_name] = [
                {
                    "session_id": record["session_id"],
                    "kind": record["kind"],
                    "scene_name": record["scene_name"],
                    "router_position": record["router_position"],
                    "action_name": record["action_name"],
                    "num_people": record["num_people"],
                    "valid_frames": record["valid_frames"],
                    "human_instances": record["human_instances"],
                }
                for record in records
            ]
        return {"split_rows": split_rows, "split_json": split_json}

    def write_outputs(self, summary: Dict[str, Any]) -> None:
        write_json(self.output_dir / SUMMARY_JSON_NAME, summary)
        self.write_summary_markdown(summary)
        write_csv(
            self.output_dir / "subject_action_stats.csv",
            summary["subject_action_stats"],
            ["subject_id", "action_name", "sessions", "valid_frames", "human_instances"],
        )
        write_csv(
            self.output_dir / "action_stats.csv",
            summary["action_stats"],
            ["action_name", "sessions", "valid_frames", "human_instances", "subjects", "scenes", "router_configurations", "subset_types"],
        )
        write_csv(
            self.output_dir / "people_distribution.csv",
            summary["people_distribution"],
            ["num_people", "frames", "human_instances"],
        )
        write_csv(
            self.output_dir / "scene_router_stats.csv",
            summary["scene_router_stats"],
            ["scene_name", "router_position", "tx_coords", "rx_coords", "sessions", "background_sessions", "valid_frames", "human_instances"],
        )
        write_csv(
            self.output_dir / "occlusion_free_stats.csv",
            summary["occlusion_free_stats"],
            ["metric", "value"],
        )
        write_csv(
            self.output_dir / "physical_occlusion_stats.csv",
            summary["physical_occlusion_stats"],
            ["occluder_type", "distance_cm", "sessions", "raw_paired_frames", "valid_paired_frames", "discarded_frames"],
        )
        write_csv(
            self.output_dir / "quality_control_stats.csv",
            summary["quality_control_stats"],
            ["metric", "value", "notes"],
        )
        write_csv(
            self.output_dir / "split_stats.csv",
            summary["split_stats"],
            ["split", "sessions", "frames", "human_instances", "source", "notes"],
        )
        split_suggestion_path = self.output_dir / "split_suggestion.json"
        write_json(split_suggestion_path, summary["split_suggestion"])
        self.write_latex_tables(summary)

    def write_summary_markdown(self, summary: Dict[str, Any]) -> None:
        counts = summary["counts"]
        lines: List[str] = []
        lines.append("# RF-Avatar Dataset Summary")
        lines.append("")
        lines.append("## Overall Counts")
        lines.append("")
        lines.append(f"- Number of subjects: {counts['number_of_subjects']}")
        lines.append(f"- Number of action categories: {counts['number_of_action_categories']}")
        lines.append(f"- Number of indoor scenes: {counts['number_of_indoor_scenes']}")
        lines.append(f"- Number of router deployment configurations: {counts['number_of_router_deployment_configurations']}")
        lines.append(f"- Number of recording sessions: {counts['number_of_recording_sessions']}")
        lines.append(f"- Number of valid RGB-D frames: {counts['number_of_valid_rgbd_frames']}")
        lines.append(f"- Number of valid CSI samples: {counts['number_of_valid_csi_samples']}")
        lines.append(f"- Number of valid BODY-38 skeleton annotations: {counts['number_of_valid_body38_skeleton_annotations']}")
        lines.append(f"- Number of valid human point cloud pseudo-labels: {counts['number_of_valid_human_point_cloud_pseudo_labels']}")
        lines.append(f"- Number of human instances: {counts['number_of_human_instances']}")
        lines.append(f"- Maximum number of people per frame: {counts['maximum_number_of_people_per_frame']}")
        lines.append(f"- Background / no-person sessions detected: {counts['background_sessions_detected']}")
        lines.append("")
        lines.append("## Parsing Rules and Assumptions")
        lines.append("")
        for rule in summary["inference_rules"]:
            lines.append(f"- {rule}")
        lines.append("")
        lines.append("## Notes on Reliability")
        lines.append("")
        for note in summary["parsing_notes"]:
            lines.append(f"- {note}")
        for note in summary["quality_notes"]:
            lines.append(f"- {note}")
        lines.append("")
        lines.append("## Manual Confirmation Needed")
        lines.append("")
        for note in summary["manual_confirmations"]:
            lines.append(f"- {note}")
        lines.append("")
        lines.append("## Synthetic Visual Occlusion Augmentation")
        lines.append("")
        lines.append("Synthetic mask augmentation is not counted as an independent dataset. It is an online training-time visual augmentation applied on top of the original RGB-D samples.")
        lines.append("")
        lines.append("- Augmentation type: online random CutOut")
        lines.append("- Applied modalities: RGB and depth")
        lines.append("- CSI changed: No")
        lines.append("- CutOut probability: p = 0.6")
        lines.append("- CutOut area ratio: 0.15 to 0.5 of image area")
        lines.append("- Aspect ratio range: 0.3 to 3.3")
        lines.append("- Mask value: 0.0")
        lines.append("")
        lines.append("Synthetic occlusion stress test:")
        lines.append("")
        lines.append("- Bottom-half occlusion ratio = 0.5")
        lines.append("- Masked modalities = RGB and depth")
        lines.append("- CSI changed = No")
        lines.append("")
        lines.append("Important: the synthetic mask only simulates visual missingness. It does not simulate real physical occlusion that changes RF propagation and therefore does not replace the paired physical-occlusion subset.")
        lines.append("")
        lines.append("## Output Files")
        lines.append("")
        for filename in [
            SUMMARY_JSON_NAME,
            SUMMARY_MD_NAME,
            "subject_action_stats.csv",
            "action_stats.csv",
            "people_distribution.csv",
            "scene_router_stats.csv",
            "occlusion_free_stats.csv",
            "physical_occlusion_stats.csv",
            "quality_control_stats.csv",
            "split_stats.csv",
            "split_suggestion.json",
            LATEX_TABLES_NAME,
        ]:
            lines.append(f"- {filename}")
        (self.output_dir / SUMMARY_MD_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def latex_escape(self, value: Any) -> str:
        text = str(value)
        return (
            text.replace("\\", "\\textbackslash{}")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("_", "\\_")
            .replace("#", "\\#")
            .replace("{", "\\{")
            .replace("}", "\\}")
        )

    def write_latex_tables(self, summary: Dict[str, Any]) -> None:
        counts = summary["counts"]
        people_distribution = summary["people_distribution"]
        scene_router_stats = summary["scene_router_stats"]
        physical_stats = summary["physical_occlusion_stats"]
        split_stats = [row for row in summary["split_stats"] if row["split"] in {"train", "val", "test"}]

        lines: List[str] = []
        lines.append("# LaTeX Table Drafts")
        lines.append("")
        lines.append("## Table I. Dataset Summary")
        lines.append("")
        lines.append("```latex")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{RF-Avatar dataset summary.}")
        lines.append("\\begin{tabular}{lr}")
        lines.append("\\toprule")
        lines.append("Statistic & Value \\\\")
        lines.append("\\midrule")
        lines.append(f"Subjects & {counts['number_of_subjects']} \\\\")
        lines.append(f"Action categories & {counts['number_of_action_categories']} \\\\")
        lines.append(f"Indoor scenes & {counts['number_of_indoor_scenes']} \\\\")
        lines.append(f"Router deployments & {counts['number_of_router_deployment_configurations']} \\\\")
        lines.append(f"Recording sessions & {counts['number_of_recording_sessions']} \\\\")
        lines.append(f"Valid RGB-D frames & {counts['number_of_valid_rgbd_frames']} \\\\")
        lines.append(f"Valid CSI samples & {counts['number_of_valid_csi_samples']} \\\\")
        lines.append(f"Valid BODY-38 annotations & {counts['number_of_valid_body38_skeleton_annotations']} \\\\")
        lines.append(f"Valid point-cloud pseudo-labels & {counts['number_of_valid_human_point_cloud_pseudo_labels']} \\\\")
        lines.append(f"Human instances & {counts['number_of_human_instances']} \\\\")
        lines.append(f"Max people / frame & {counts['maximum_number_of_people_per_frame']} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("```")
        lines.append("")
        lines.append("## Table II. Multi-person Distribution")
        lines.append("")
        lines.append("```latex")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Distribution by people-count.}")
        lines.append("\\begin{tabular}{lrr}")
        lines.append("\\toprule")
        lines.append("People count & Frames & Human instances \\\\")
        lines.append("\\midrule")
        for row in people_distribution:
            lines.append(f"{row['num_people']}-person & {row['frames']} & {row['human_instances']} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("```")
        lines.append("")
        lines.append("## Table III. Router-position Distribution")
        lines.append("")
        lines.append("```latex")
        lines.append("\\begin{table*}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Scene and router deployment distribution.}")
        lines.append("\\begin{tabular}{llllrrr}")
        lines.append("\\toprule")
        lines.append("Scene & Router pos. & Tx coords & Rx coords & Sessions & Frames & Human inst. \\\\")
        lines.append("\\midrule")
        for row in scene_router_stats:
            lines.append(
                f"{self.latex_escape(row['scene_name'])} & {self.latex_escape(row['router_position'])} & "
                f"{self.latex_escape(row['tx_coords'])} & {self.latex_escape(row['rx_coords'])} & "
                f"{row['sessions']} & {row['valid_frames']} & {row['human_instances']} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")
        lines.append("```")
        lines.append("")
        lines.append("## Table IV. Physical-occlusion Dataset")
        lines.append("")
        lines.append("```latex")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Paired physical-occlusion subset statistics.}")
        lines.append("\\begin{tabular}{lrrrrr}")
        lines.append("\\toprule")
        lines.append("Occluder & Dist. (cm) & Sessions & Raw pairs & Valid pairs & Discarded \\\\")
        lines.append("\\midrule")
        for row in physical_stats:
            lines.append(
                f"{self.latex_escape(row['occluder_type'])} & {row['distance_cm']} & {row['sessions']} & "
                f"{row['raw_paired_frames']} & {row['valid_paired_frames']} & {row['discarded_frames']} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("```")
        lines.append("")
        lines.append("## Table V. Evaluation Protocols")
        lines.append("")
        lines.append("```latex")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Suggested session-level evaluation protocol.}")
        lines.append("\\begin{tabular}{lrrr}")
        lines.append("\\toprule")
        lines.append("Split & Sessions & Frames & Human instances \\\\")
        lines.append("\\midrule")
        for row in split_stats:
            lines.append(f"{row['split'].title()} & {row['sessions']} & {row['frames']} & {row['human_instances']} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("```")
        (self.output_dir / LATEX_TABLES_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    if not data_root.exists():
        print(f"[ERROR] data root does not exist: {data_root}", file=sys.stderr)
        return 1
    analyzer = DatasetAnalyzer(
        data_root=data_root,
        output_dir=output_dir,
        check_finite_arrays=bool(args.check_finite_arrays),
    )
    analyzer.run()
    print(f"[OK] Dataset statistics written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

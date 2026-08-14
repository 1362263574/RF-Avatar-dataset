# Paper Table Notes

## Key Notes

- All-modality valid sample count is not directly available from the current summary files.
- Table II uses the effective people count from the valid skeleton-derived distribution, not the raw directory subject token count.
- One recorded router configuration appears only in background/no-person sessions and is not counted as a human reconstruction configuration unless explicitly stated.
- 3 recorded router configuration(s) currently contribute recorded sessions without retained multimodal human samples under the present filtering rules.
- The WiFi receiver is co-located with the ZED-X stereo camera optical center and is therefore defined as the origin of the camera coordinate system.
- The transmitter position is represented by a manually measured 3D offset relative to the receiver/camera origin.
- Since the receiver is aligned with the stereo camera optical center, the receiver coordinate is not separately stored in the metadata. The camera coordinate origin is used as the receiver coordinate.
- The `nothing` and `newlab_nothing` clips are background calibration captures and should be interpreted as background/baseline measurements rather than human action categories.
- All splits should be performed at the recording-session level to avoid temporal leakage between training and testing.
- Raw action categories are parsed from directory or metadata names and may include occlusion-related suffixes.
- The current dataset summary reports subject-count and skeleton-count mismatch frames; these require manual confirmation before writing a stronger claim about annotation completeness.
- Quality-control statistics are kept for appendix or internal reporting because several fields such as ghost skeleton removal and NaN/Inf checks are unavailable or not checked.
- Final retained samples in the QC table should not be interpreted as all-modality valid samples.

## Detected Consistency Checks

- Physical-occlusion total valid paired frames from CSV = 1980.
- The expected reference count from perfect_train_list.json is 1980.
- Match status = matched.
- Background-only router configuration detected = Yes.
- Additional recorded configurations without retained multimodal human samples = 3.
- Subject-count and skeleton-count mismatch mentioned in summary = Yes.
- Draft split RGB-D frame total = 7972 versus dataset valid RGB-D frames = 11331.
- Draft split human-instance total = 11510 versus dataset human instances = 9160.
- Current split statistics have inconsistent human-instance totals and should not be used in the paper until recomputed.

## Source Annotations

### Parsing Notes
- Loaded perfect_train_list.json with 1980 paired entries for physical occlusion statistics.
- No usable RFAvatarDataset loader was found in the current workspace; all statistics are filesystem-derived.
- 4 router configuration(s) were observed only in zero-valid-human sessions; keep this in mind if the paper should report human-data-only router deployments.

### Manual Confirmations
- 1022 valid frames had a mismatch between subject-count parsed from the directory name and people-count parsed from pose_3d_gt.json.
- Confirm whether action names should follow directory names (e.g., sit_occlusion_lower) or clip_meta.json values when the two differ slightly.

### Reused Stats Artifacts

- dataset_summary.md and latex_tables.md were read as existing references but the paper_tables outputs were freshly regenerated from the structured CSV/JSON stats.
- The previous latex_tables.md was not edited in place.

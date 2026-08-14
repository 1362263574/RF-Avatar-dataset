# RF-Avatar Dataset Summary

## Overall Counts

- Number of subjects: 12
- Number of action categories: 30
- Number of indoor scenes: 3
- Number of router deployment configurations: 9
- Number of recording sessions: 924
- Number of valid RGB-D frames: 11331
- Number of valid CSI samples: 8009
- Number of valid BODY-38 skeleton annotations: 6987
- Number of valid human point cloud pseudo-labels: 3838
- Number of human instances: 9160
- Maximum number of people per frame: 3
- Background / no-person sessions detected: 4

## Parsing Rules and Assumptions

- Subject IDs are parsed from directory tokens matching K<number> or U<number>; e.g., K1K3 or U1U2 expands to individual subjects, and each token is treated as a distinct person identity.
- subject_action_stats.csv attributes each valid frame in a multi-person session to every subject token parsed from that session directory.
- action_stats.csv aggregates each action over all sessions and reports valid frame count, human instance count, covered subjects, scenes, and router configurations.
- router_position is parsed from the scene-router directory suffix (e.g., Room_Pos1_L20 -> Pos1_L20).
- scene_name is normalized from clip_meta.json plus the scene-router directory prefix, so obvious typos such as romm/newlba are folded back to room/newlab.
- Regular-subset scanning includes the legacy 无遮挡 root plus any additional top-level *_row directories that match the same session layout.
- The WiFi receiver is co-located with the ZED-X camera optical center and is therefore treated as the camera coordinate origin; the transmitter position is represented by geometry_calibration.tx_antenna_offset_mm relative to that origin.
- Physical-occlusion valid paired frames prefer perfect_train_list.json; discarded paired frames are estimated as filesystem raw pairs minus valid perfect-list pairs.

## Notes on Reliability

- Loaded perfect_train_list.json with 1980 paired entries for physical occlusion statistics.
- No usable RFAvatarDataset loader was found in the current workspace; all statistics are filesystem-derived.
- 4 router configuration(s) were observed only in zero-valid-human sessions; keep this in mind if the paper should report human-data-only router deployments.
- NaN / Inf sample counting was left as unknown because --check-finite-arrays was not enabled and depth maps routinely contain NaNs as valid missing-depth markers.

## Manual Confirmation Needed

- 1022 valid frames had a mismatch between subject-count parsed from the directory name and people-count parsed from pose_3d_gt.json.
- Confirm whether background 'nothing' sessions should be reported in the paper as auxiliary no-person negatives or excluded entirely from Experimental Setup tables.
- Confirm whether action names should follow directory names (e.g., sit_occlusion_lower) or clip_meta.json values when the two differ slightly.

## Synthetic Visual Occlusion Augmentation

Synthetic mask augmentation is not counted as an independent dataset. It is an online training-time visual augmentation applied on top of the original RGB-D samples.

- Augmentation type: online random CutOut
- Applied modalities: RGB and depth
- CSI changed: No
- CutOut probability: p = 0.6
- CutOut area ratio: 0.15 to 0.5 of image area
- Aspect ratio range: 0.3 to 3.3
- Mask value: 0.0

Synthetic occlusion stress test:

- Bottom-half occlusion ratio = 0.5
- Masked modalities = RGB and depth
- CSI changed = No

Important: the synthetic mask only simulates visual missingness. It does not simulate real physical occlusion that changes RF propagation and therefore does not replace the paired physical-occlusion subset.

## Output Files

- dataset_summary.json
- dataset_summary.md
- subject_action_stats.csv
- action_stats.csv
- people_distribution.csv
- scene_router_stats.csv
- occlusion_free_stats.csv
- physical_occlusion_stats.csv
- quality_control_stats.csv
- split_stats.csv
- split_suggestion.json
- latex_tables.md

# RF-Avatar Paper Dataset Tables

## Dataset Paragraph

We collect a synchronized RGB-D and WiFi CSI dataset from 12 subjects performing 30 action categories across 3 indoor scenes and 9 recorded router deployment configurations. Among these router configurations, 5 contain valid human samples and 1 is used for background/no-person calibration. The remaining 3 configurations currently contribute recorded sessions without retained multimodal human samples under the present filtering rules. The WiFi receiver is co-located with the ZED-X camera optical center and is therefore defined as the origin of the camera coordinate system, while the transmitter position is represented by a manually measured 3D offset relative to this receiver/camera origin. The dataset contains 11331 valid RGB-D frames, 8009 valid CSI samples, 6987 skeleton-valid frames, and 9160 human instances, with up to 3 people appearing in one frame. After depth-based pseudo-label generation and quality control, 3838 reliable human-centric point cloud pseudo-labels are retained for point cloud supervision. In addition, we construct a paired physical-occlusion subset containing 1980 valid occluded--unoccluded paired samples with three occluder materials and three occluder distances.

## Synthetic Visual Occlusion Augmentation

Synthetic visual occlusion is not an independently collected dataset. During training, we apply online random CutOut to RGB and depth observations with probability 0.6. The mask area is randomly sampled from 15% to 50% of the image area, and the aspect ratio is sampled from 0.3 to 3.3. The masked values are set to 0.0. The CSI stream is kept unchanged, since this augmentation only simulates visual missingness and does not model RF propagation changes caused by real physical occluders. For the synthetic occlusion stress test, we mask the lower 50% of RGB-D observations while keeping the original CSI unchanged. The synthetic mask only simulates visual missingness and does not simulate RF propagation shifts.

## Table I. Dataset Summary

| Item | Value |
| --- | --- |
| Subjects | 12 |
| Action categories | 30 |
| Indoor scenes | 3 |
| Recorded router deployment configurations | 9 |
| Human reconstruction router configurations | 5 |
| Background/no-person router configurations | 1 |
| Additional recorded configurations without retained multimodal human samples | 3 |
| Recording sessions | 924 |
| Valid RGB-D frames | 11331 |
| Valid CSI samples | 8009 |
| Skeleton-valid frames | 6987 |
| Point-cloud-supervised samples | 3838 |
| Human instances | 9160 |
| Maximum people per frame | 3 |
| Physical-occlusion paired samples | 1980 |

## Table II. Multi-person Distribution

| Number of people | Frames | Human instances |
| --- | --- | --- |
| 1-person | 5238 | 5238 |
| 2-person | 1325 | 2650 |
| 3-person | 424 | 1272 |
| Total | 6987 | 9160 |

Note: The multi-person distribution is computed over skeleton-valid frames. The number of people per frame is determined by valid BODY-38 skeleton instances rather than directory-level subject tokens.

## Table III. Scene and Router-position Distribution

| Scene | Router position | Config ID | Sessions | Valid frames | Human instances | Type |
| --- | --- | --- | --- | --- | --- | --- |
| Lab | Pos0 | RC1 | 163 | 1895 | 2744 | Human |
| Lab | Pos0_Unseen | RC2 | 107 | 0 | 0 | No retained human samples |
| Lab | Pos1_R20 | RC3 | 82 | 926 | 1480 | Human |
| Lab | Pos1_R20_Unseen | RC4 | 79 | 0 | 0 | No retained human samples |
| NewLab | Pos2 | RC5 | 243 | 1980 | 1980 | Human |
| NewLab | Pos2 | RC6 | 1 | 0 | 0 | Background |
| Room | Pos0 | RC7 | 138 | 1527 | 2112 | Human |
| Room | Pos0_Unseen | RC8 | 48 | 0 | 0 | No retained human samples |
| Room | Pos1_L20 | RC9 | 63 | 659 | 844 | Human |

Note: For all router configurations, the WiFi receiver is aligned with the ZED-X camera optical center and is treated as the camera coordinate origin. The transmitter position is represented by a manually measured offset relative to this receiver/camera origin. One recorded router configuration appears only in background/no-person sessions and is not counted as a human reconstruction configuration. In addition, 3 recorded configurations contain recorded sessions but no retained multimodal human samples under the current filtering rules. 

## Table IV. Physical-occlusion Dataset

| Occluder | Distance | Sessions | Raw paired frames | Valid paired frames | Discarded frames |
| --- | --- | --- | --- | --- | --- |
| Black cloth | 100 cm | 28 | 401 | 255 | 146 |
| Black cloth | 150 cm | 27 | 377 | 160 | 217 |
| Black cloth | 200 cm | 27 | 380 | 123 | 257 |
| Cardboard board | 100 cm | 27 | 365 | 197 | 168 |
| Cardboard board | 150 cm | 27 | 362 | 130 | 232 |
| Cardboard board | 200 cm | 27 | 363 | 181 | 182 |
| Foam board | 100 cm | 27 | 364 | 348 | 16 |
| Foam board | 150 cm | 27 | 370 | 326 | 44 |
| Foam board | 200 cm | 27 | 362 | 260 | 102 |
| Total | - | 244 | 3344 | 1980 | 1364 |

Note: Valid paired frames denote occluded--unoccluded pairs retained after quality control.

## Table V. Evaluation Protocols

| Protocol | Training data | Testing data | Purpose |
| --- | --- | --- | --- |
| Standard reconstruction | Occlusion-free multimodal data | Clean test sessions | Evaluate basic multi-person 3D skeleton and point cloud reconstruction |
| Multi-person reconstruction | 1--3 person data | Test sessions grouped by the number of people | Evaluate multi-person instance decoupling |
| Synthetic visual occlusion | Online CutOut-augmented RGB-D with unchanged CSI | Masked RGB-D generated from clean test sessions | Test robustness to visual missingness |
| Physical occlusion | Paired occluded/unoccluded recordings | Real physical-occlusion sessions | Evaluate robustness to real occluders and RF propagation shifts |
| Cross-router-position generalization | Selected router configurations | Held-out or moved router configurations | Evaluate deployment generalization with router coordinates |

Note: Synthetic visual occlusion is not an independently collected dataset; it is generated online or during stress testing from clean RGB-D samples, while CSI remains unchanged.

## Table VI. Train / Validation / Test Split (Draft / Not for Main Paper)

| Split | Sessions | RGB-D frames | Skeleton-valid frames | Human instances | Physical-occlusion pairs |
| --- | --- | --- | --- | --- | --- |
| Train | 412 | 5630 | Not recomputed | 8091 | 1403 |
| Val | 88 | 1184 | Not recomputed | 1705 | 299 |
| Test | 91 | 1158 | Not recomputed | 1714 | 278 |

Draft note: Current split statistics have inconsistent human-instance totals and should not be used in the paper until recomputed. All splits should be performed at the recording-session level to avoid temporal leakage between training and testing.

## Appendix Table A. Subject-level Data Distribution

| Subject | Action categories | Sessions | Valid frames | Human instances |
| --- | --- | --- | --- | --- |
| K1 | 28 | 166 | 1964 | 1964 |
| K2 | 23 | 91 | 970 | 970 |
| K3 | 28 | 268 | 3098 | 3098 |
| K4 | 28 | 153 | 1829 | 1829 |
| K5 | 19 | 66 | 766 | 766 |
| K6 | 19 | 47 | 533 | 533 |

Note: In multi-person sessions, a recording session may be attributed to multiple participating subjects; therefore, subject-level totals are not expected to sum to the dataset-level frame count. Raw action categories are parsed from directory or metadata names and may include occlusion-related suffixes.

## Appendix Table B. Quality Control Statistics

| Item | Count / Status |
| --- | --- |
| Raw samples | 11331 |
| Valid RGB-D frames | 11331 |
| Valid CSI samples | 8009 |
| Skeleton-valid frames | 6987 |
| Valid point cloud pseudo-labels | 3838 |
| Human instances | 9160 |
| Missing RGB-D | 0 |
| Invalid depth | 0 |
| Invalid CSI | 3322 |
| Missing skeleton | 0 |
| Incomplete skeleton | 6699 |
| Ghost skeleton removed | unknown |
| Invalid point cloud | 0 |
| Shape error | 0 |
| NaN / Inf samples | Not checked |
| Final retained samples | 8009 |


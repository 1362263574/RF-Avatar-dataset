# RF-Avatar Dataset

This repository is the public landing page for the **RF-Avatar dataset**.  
It provides the dataset download entry, preview images, dataset statistics, paper-ready tables, and helper scripts.

The raw dataset is **not hosted directly on GitHub** because of repository size limits.  
Users should download the dataset from the external storage link below.

## Preview

### Sample Scene 1

![RF-Avatar sample scene 1](assets/sample_lab_pos0_rgb.png)

### Sample Scene 2

![RF-Avatar sample scene 2](assets/sample_room_pos0_rgb.png)

## Download

Current public release:

- Subset: `Occlusion-free subset`
- Notes: the previously separated `new_data_row` content has already been merged into the latest occlusion-free release
- Download link: [https://pan.quark.cn/s/ec0b5a6b0257](https://pan.quark.cn/s/ec0b5a6b0257)
- Extraction code: `RUQ4`

You can also copy the following text into the Quark Drive app:

```text
Shared folder: Occlusion-free subset
Link: https://pan.quark.cn/s/ec0b5a6b0257
Extraction code: RUQ4
```

## Repository Contents

This upload package contains:

- `README.md`: public dataset page
- `.gitignore`: safe ignore rules for local development
- `assets/`: preview images used in the README
- `dataset_stats/`: dataset statistics files
- `paper_tables/`: paper-ready experiment tables
- `scripts/`: reusable analysis and table-generation scripts

## Recommended Release Strategy

This repository is intended to work as:

1. A GitHub landing page for the dataset
2. A download entry point for visitors
3. A place for statistics, tables, and helper scripts

The raw multimodal dataset should continue to be distributed through external storage rather than committed to GitHub.

## Suggested Structure

```text
RF-Avatar-dataset/
├─ README.md
├─ .gitignore
├─ assets/
├─ dataset_stats/
├─ paper_tables/
└─ scripts/
```

## Notes

- If you later release additional subsets, add new download sections to this README.
- If you prepare a paper release, consider adding `LICENSE`, `CITATION.cff`, and a more formal dataset card.
- If you do not want to show clear RGB images on the GitHub front page, you can replace the preview images with depth, skeleton, or anonymized examples.


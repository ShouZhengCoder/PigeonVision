# Teacher Dataset Report

## Location

- Raw archives: `data/raw_from_teacher/`
- Extracted dataset: `data/extracted/`
- Main metadata directory: `data/extracted/datasetXGN/`
- Duplicate metadata directory `data/extracted/metadataXG/` was verified and then removed from the extracted dataset to avoid redundancy. The original `metaXG.zip` archive is still preserved in `data/raw_from_teacher/`.

## Overall Scale

- Total image files: 31,896
- Image directories: `1` through `12`
- Images per directory: 2,658
- Corrupt/unreadable images found: 0
- Unique image IDs: 31,896
- Duplicate image IDs across image directories: 0

## Image Format And Size

- Image modes:
  - RGB: 31,222
  - RGBA: 363
  - P: 250
  - CMYK: 60
  - L: 1
- Unique image sizes: 5,434
- Most common sizes include:
  - 1200x1000: 691
  - 750x530: 682
  - 800x600: 573
  - 1000x750: 548
  - 1000x833: 469

Recommendation: convert images to RGB during training or preprocessing, because the dataset contains RGBA, palette, CMYK, and grayscale files.

## `pigeon.csv`

Path: `data/extracted/datasetXGN/pigeon.csv`

Fields:

- `ID`
- `PID`
- `CID`
- `SID`
- `NAME`
- `COLOR`
- `EYE`
- `PG_ID`
- `SEX`
- `BLOOD`
- `IMG`

Observed rows:

- Total rows: 113,844
- Unique `ID` values: 113,844
- Rows whose `ID` has a local image: 27,854
- CSV rows without a local image: 85,990
- Local images without a `pigeon.csv` row: 4,042

Important implication: `pigeon.csv` is larger than this local image subset. Do not assume every CSV row has a downloaded image, and do not assume every local image has full tabular metadata.

Top metadata values:

- `COLOR`: `灰`, `--`, `雨点`, `绛`, `花`, `灰白条`
- `EYE`: `黄眼`, `砂眼`, empty, `黄`, `砂`, `牛眼`
- `SEX`: mostly `2`, `1`, and `0`, but there are malformed values in a small number of rows
- `BLOOD`: many empty values; common non-empty examples include `桑杰士`, `詹森`, `胡本`, `盖比`, `速霸龙`

## Detection Annotations

Primary path: `data/extracted/datasetXGN/anotations/`

Notes:

- The directory name is misspelled as `anotations`.
- Before removal, `data/extracted/metadataXG/anotations/` contained the same 9,979 annotation JSON files with identical content.
- This identical-content statement applies to the annotation JSON files only, not to the original zip archives as byte-for-byte identical files.
- Each JSON normally has:
  - `img`: image filename
  - `height`: image height
  - `weidth`: image width, misspelled
  - `bbs`: list of bounding boxes
  - each bounding box has `label` and `bbx`

Coverage:

- Annotation JSON files: 9,979
- Annotated images present locally: 9,979
- Local images without annotation JSON: 21,917

Annotation coverage by image directory:

- `1`: 0 / 2,658
- `2`: 0 / 2,658
- `3`: 0 / 2,658
- `4`: 0 / 2,658
- `5`: 0 / 2,658
- `6`: 2,658 / 2,658
- `7`: 0 / 2,658
- `8`: 0 / 2,658
- `9`: 0 / 2,658
- `10`: 2,005 / 2,658
- `11`: 2,658 / 2,658
- `12`: 2,658 / 2,658

Label counts:

- `eye`: 9,535
- `900`: 410
- `mouse`: 2

Bounding boxes per annotation JSON:

- 0 boxes: 176 files
- 1 box: 9,675 files
- 2 boxes: 116 files
- 3 boxes: 9 files
- 4 boxes: 2 files
- 5 boxes: 1 file

Potentially invalid boxes:

- 13 bounding boxes have zero/negative width or height, or coordinates outside image bounds.
- Examples: `126918.json`, `91768.json`, `390149.json`, `110175.json`, `144730.json`, `258223.json`.

Recommendation: before training a detector, filter invalid boxes and decide whether labels `900` and `mouse` should be remapped or excluded. The dominant usable detection class appears to be `eye`.

## Genealogy / Bloodline Files

`relations.csv`:

- Rows: 250,207
- Unique bloodline IDs: 81,751
- Unique image IDs referenced: 46,788

Verified difference between extracted `datasetXGN` and removed `metadataXG`:

- `blood.csv`, `city.json`, `city_list.json`, `details.txt`, `img_list.txt`, `pigeon.csv`, and `readme.txt` are byte-identical after extraction.
- `metadataXG/relations.csv` has one extra header row: `blood_id,IMG`.
- After ignoring that header row, the relation rows are the same as `datasetXGN/relations.csv`.
- The original zip sizes differ because the compressed byte streams and `relations.csv` contents differ, even though the annotation JSON files are identical.

`blood.csv`:

- Rows: 28,910
- Each row maps one bloodline ID to one or more image IDs.
- Some bloodline IDs are highly connected:
  - `B98-3158062`: 525 image IDs
  - `B06-3008003`: 393 image IDs
  - `B01-6455003`: 326 image IDs

Important implication: the bloodline graph references more image IDs than are present in the local image subset. Treat it as a broader metadata graph, not only as labels for the 31,896 local images.

## Text Metadata

`details.txt` maps image IDs to long Chinese text descriptions. These descriptions often include parentage, pedigree, race performance, names, and bloodline references.

This file is useful for:

- text mining
- pedigree relation extraction
- weak supervision
- multimodal image-text experiments

It is not clean enough to use directly as structured labels without preprocessing.

## Suitable ML Tasks

Good fits:

- Eye-region object detection using the annotated subset.
- Pigeon image classification by available metadata fields such as color, eye type, sex, region, or bloodline.
- Multimodal retrieval: image plus text description.
- Pedigree/bloodline graph analysis using `relations.csv` and `blood.csv`.

Risky without cleaning:

- Full-dataset object detection, because only 9,979 of 31,896 images have annotation JSON files.
- Fine-grained bloodline classification, because `BLOOD` has many missing and highly imbalanced values.
- Direct use of `SEX`, because a small number of rows contain malformed values.

## Practical Preprocessing Plan

1. Build a manifest table keyed by image ID and local file path.
2. Left-join `pigeon.csv` onto local image IDs.
3. Add annotation path and valid bounding boxes where available.
4. Normalize all images to RGB when loading.
5. Clean categorical labels:
   - map `SEX` to known values only: `0`, `1`, `2`
   - normalize eye values such as `黄`, `黃眼`, `黄眼`
   - normalize color aliases only after manual review
6. For detection training:
   - use directories `6`, `10`, `11`, and `12`
   - drop empty annotation files or keep them as negative samples intentionally
   - filter 13 invalid boxes
   - decide whether to keep only `eye`

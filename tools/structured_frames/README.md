# Keystep Extraction Guide

This directory contains the keystep extraction programs currently used in StructVLA.

## Recommended Program

Use the unified extraction program by default:

- `structured_frames_extract.py`


Explicit command-line parameters always take priority over preset values.

## Simpler

Recommended command:

```bash
python tools/structured_frames/structured_frames_extract.py   --datasets simpler   --dataset /path/to/your.pkl   --out_dir /path/to/output
```

## LIBERO

Recommended command:

```bash
python tools/structured_frames/structured_frames_extract.py   --datasets libero   --dataset /path/to/your.pkl   --out_dir /path/to/output
```

## Real-World Data

For real-world teleoperation data, you can also use the same unified program.
When `--datasets real` is provided, the extractor runs with the generic default parameters unless you explicitly override them.

Example:

```bash
python tools/structured_frames/structured_frames_extract.py   --datasets real   --dataset /remote-home/share/real-data/meta/real_all_norm.pkl   --out_dir /remote-home/structvla/documents/keysteps_real_dataset
```

This will write results to:

- `/remote-home/structvla/documents/keysteps_real_dataset/triplets_manifest.csv`
- `/remote-home/structvla/documents/keysteps_real_dataset/summary.json`

## Real-World Specialized Script

In practice, real-world teleoperation data is usually clean enough that a simpler keystep extraction rule is sufficient.
For that reason, we also keep a dedicated real-world extractor:

- `extract_real_robot.py`

This script uses a more direct logic specialized for real-world data. In most cases, the default parameters are enough.

Example:

```bash
python tools/structured_frames/extract_real_robot.py   --dataset /remote-home/share/real-data/meta/real_all_norm.pkl   --out_dir /remote-home/structvla/documents/extract_real_robot
```

## Further Filtering of Structured Frames

After the initial candidate structured frames are extracted and saved in the keystep CSV file, we also provide an optional lightweight VLM-based filtering step for further refinement. This stage can be used when additional filtering is desired, its purpose is to remove clearly unstable or visually redundant candidates and make the final selected frames more reliable.

The filtering program is:

- `VLM_filter.py`

Example:

```bash
python tools/structured_frames/VLM_filter.py \
  --manifest_csv /path/to/triplets_manifest.csv \
  --dataset_pkl /path/to/dataset.pkl \
  --raw_image_root /path/to/raw_image_root \
  --out_csv /path/to/triplets_manifest_filtered.csv \
  --out_debug_dir /path/to/vlm_filter_debug \
  --api_key $OPENAI_API_KEY
```

This program takes the extracted keystep CSV as input and writes:

- a filtered CSV specified by `--out_csv`
- optional per-episode debug JSON files and `filter_summary.json`

## Output Files

The extractor writes:

- `triplets_manifest.csv`: the main keystep manifest used by downstream training or filtering.
- `summary.json`: a lightweight run summary including dataset path, selected mode, and total triplet count.

## Notes

- For new datasets, start with `structured_frames_extract.py` and choose a dataset name through `--datasets`.
- If the new dataset is structurally closer to real-world teleoperation data, `extract_real_robot.py` can also be a better baseline.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLM-based offline filter for keystep CSV.

Input:
  - original triplets_manifest.csv
  - original Bridge-style PKL dataset
  - raw RGB image root directory

Output:
  - filtered triplets_manifest_filtered.csv
  - optional per-episode JSON debug outputs

Behavior:
  - group candidates by episode_id
  - recover raw RGB image paths from the PKL frame indices and the raw image root
  - provide VLM with:
      * task instruction
      * candidate csv row ids
      * corresponding candidate images
  - VLM only filters:
      * unstable perturbation points
      * task-incoherent transient false positives
      * overly dense redundant local points
  - VLM returns kept row ids
  - final CSV is created by selecting original rows only, preserving the original schema
"""

import os
import io
import re
import glob
import json
import base64
import pickle
import argparse
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai>=1.0.0")

# ---------------------------------------------------------------------
# Prompt and schema
# ---------------------------------------------------------------------

SYSTEM_FILTER_IMG = """\
You are filtering candidate structured frames for a robot manipulation episode using images as the primary source of truth.

The input is an ordered list of candidate keysteps from the original CSV file, extracted from robot-centric kinematic cues. Each candidate is identified by its row index in the original CSV file, together with its image and the task instruction.

Your task is only to perform a conservative filtering of the candidates. Do not rewrite the instruction, do not add new events, and do not reconstruct the sequence.

Filtering goals:
1) Remove clearly unstable perturbation points caused by control noise, hardware jitter, or transient fluctuations.
2) Remove temporary false positives that clearly do not reflect a reasonable task progression.
3) Remove candidates that are visually too close to nearby neighbors and provide almost no additional information.

Important principles:
- Be conservative in filtering. Prefer keeping a candidate unless it is clearly abnormal or clearly redundant.
- When in doubt, keep the candidate rather than remove it.
- Preserve sufficient temporal coverage of the episode. The remaining candidates should still reflect the main progression of the manipulation process.
- Do not over-prune the sequence. In most normal cases, the filtered sequence should still contain multiple keyframes rather than collapsing to a single frame.
- Only remove a candidate when there is clear visual evidence that it is a noise point, a brief disturbance, or nearly duplicate with a nearby retained frame.
- If several nearby candidates correspond to essentially the same stable state, keep the earlier one.
- Preserve the original temporal order of the remaining candidates.
- Only delete candidates; do not add new ones.
- Use the images as the primary evidence.
- The task instruction is provided only as global context.
- Do not apply extra semantic rules beyond visible task coherence and local temporal consistency.

Return a JSON object with:
- filtered_ok: bool
- problems: [str, ...]
- kept_rows: [int, ...]
"""

FILTER_SCHEMA = {
    "name": "keystep_filter",
    "schema": {
        "type": "object",
        "properties": {
            "filtered_ok": {"type": "boolean"},
            "problems": {
                "type": "array",
                "items": {"type": "string"},
            },
            "kept_rows": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["filtered_ok", "problems", "kept_rows"],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _make_client(api_key: Optional[str] = None) -> OpenAI:
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", None)
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set and api_key is None.")

    try:
        return OpenAI(api_key=api_key)
    except ImportError as e:
        raise ImportError(
            "Failed to initialize OpenAI client. "
            "If you are using a SOCKS proxy, install `httpx[socks]` or `socksio`. "
            "If you do not need a proxy, unset HTTP_PROXY / HTTPS_PROXY / ALL_PROXY first."
        ) from e

def _load_and_downscale_image_as_data_url(img_path: str, max_side: int = 448) -> str:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def resolve_raw_image_path_from_code_path(code_path: str, raw_image_root: str) -> str:
    """
    Map code path like:
      /.../bridge_orig_codes_256/1/images/011.npy
    to raw image path under:
      raw_image_root/1/images/0011.jpg

    Strategy:
      1) parse episode dir and numeric frame id from code path
      2) search under raw_image_root/<episode>/images
      3) try common zero-padding patterns and image extensions
    """
    norm_path = os.path.normpath(code_path)
    parts = norm_path.split(os.sep)

    if len(parts) < 3:
        raise ValueError(f"Unexpected code path format: {code_path}")

    filename = parts[-1]
    img_dir_name = parts[-2]
    episode_dir = parts[-3]

    if img_dir_name != "images":
        raise ValueError(f"Expected 'images' directory in code path, got: {code_path}")

    stem, ext = os.path.splitext(filename)
    if ext.lower() != ".npy":
        raise ValueError(f"Expected .npy code path, got: {code_path}")

    m = re.search(r"(\d+)$", stem)
    if m is None:
        raise ValueError(f"Cannot parse numeric frame id from code path: {code_path}")

    frame_idx = int(m.group(1))
    raw_img_dir = os.path.join(raw_image_root, episode_dir, "images")
    if not os.path.isdir(raw_img_dir):
        raise FileNotFoundError(f"Raw image directory not found: {raw_img_dir}")

    candidates = []
    for width in [4, 5, 6, len(m.group(1)), 3]:
        num_str = f"{frame_idx:0{width}d}"
        for ext2 in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            candidates.append(os.path.join(raw_img_dir, num_str + ext2))

    for p in candidates:
        if os.path.exists(p):
            return p

    patterns = []
    for ext2 in ["jpg", "jpeg", "png", "bmp", "webp"]:
        patterns.append(os.path.join(raw_img_dir, f"*{frame_idx}.{ext2}"))
        patterns.append(os.path.join(raw_img_dir, f"{frame_idx}.{ext2}"))
        patterns.append(os.path.join(raw_img_dir, f"{frame_idx:04d}.{ext2}"))
        patterns.append(os.path.join(raw_img_dir, f"{frame_idx:05d}.{ext2}"))
        patterns.append(os.path.join(raw_img_dir, f"{frame_idx:06d}.{ext2}"))

    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))

    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matches = sorted(matches, key=lambda x: (len(os.path.basename(x)), x))
        return matches[0]

    raise FileNotFoundError(
        f"Cannot resolve raw image path for code path {code_path} under {raw_img_dir}"
    )

# ---------------------------------------------------------------------
# Rebuild frame-level metadata from PKL
# ---------------------------------------------------------------------

def _resolve_instruction(ep: Dict[str, Any], epi: int) -> str:
    candidate_keys = [
        "language_instruction",
        "instruction",
        "task_instruction",
        "lang",
        "language",
        "task",
        "task_name",
        "episode_name",
        "name",
    ]
    for k in candidate_keys:
        v = ep.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return f"episode_{epi}"

def build_global_frame_index_from_pkl(
    pkl_path: str,
    raw_image_root: str,
    debug_print_limit: int = 10,
) -> Dict[int, Dict[str, Any]]:
    """
    Rebuild global frame indexing in the same order as the original extractor:
      episode 0 all frames, then episode 1 all frames, ...
    Use PKL-stored code path only as an index reference, and resolve the
    corresponding raw RGB image path from raw_image_root.
    """
    with open(pkl_path, "rb") as f:
        episodes = pickle.load(f)

    if not isinstance(episodes, list):
        raise ValueError("Expected PKL to contain a list of episodes.")

    frame_map: Dict[int, Dict[str, Any]] = {}
    global_idx = 0
    printed = 0

    for epi, ep in enumerate(episodes):
        actions = ep["action"]
        imgs = ep.get("image", None)
        if imgs is None:
            raise ValueError(f"Episode {epi} does not contain 'image' field.")
        instruction = _resolve_instruction(ep, epi)

        T = len(actions)
        if len(imgs) < T:
            raise ValueError(f"Episode {epi}: number of images < number of actions.")

        for t in range(T):
            code_path = imgs[t]
            raw_image_path = resolve_raw_image_path_from_code_path(code_path, raw_image_root)

            if printed < debug_print_limit:
                print(f"[DEBUG][MAP] global_idx={global_idx} | episode={epi} | step={t}")
                print(f"[DEBUG][MAP] code_path -> {code_path}")
                print(f"[DEBUG][MAP] raw_path  -> {raw_image_path}")
                printed += 1

            frame_map[global_idx] = {
                "episode_id": str(epi),
                "step": t,
                "code_path": code_path,
                "image_path": raw_image_path,
                "instruction": instruction,
            }
            global_idx += 1

    print(f"[INFO] Built frame map with {len(frame_map)} global frames.")
    return frame_map

# ---------------------------------------------------------------------
# Build episode-level VLM input
# ---------------------------------------------------------------------

def build_episode_candidates(
    df_ep: pd.DataFrame,
    frame_map: Dict[int, Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns:
      - instruction
      - candidates: list of dicts with csv_row_id / idx_keystep / step_keystep / image_path ...
    """
    candidates: List[Dict[str, Any]] = []
    instruction: Optional[str] = None
    instructions_seen = set()

    df_ep = df_ep.sort_values(["step_keystep", "idx_keystep"]).copy()

    for _, row in df_ep.iterrows():
        csv_row_id = int(row["csv_row_id"])
        idx_keystep = int(row["idx_keystep"])

        if idx_keystep not in frame_map:
            raise KeyError(f"idx_keystep={idx_keystep} not found in PKL-rebuilt frame map.")

        meta = frame_map[idx_keystep]
        img_path = meta["image_path"]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Resolved raw image path not found: {img_path}")

        instructions_seen.add(meta["instruction"])
        if instruction is None:
            instruction = meta["instruction"]

        candidates.append({
            "csv_row_id": csv_row_id,
            "episode_id": str(row["episode_id"]),
            "idx_keystep": idx_keystep,
            "step_keystep": int(row["step_keystep"]),
            "keystep_type": str(row["keystep_type"]),
            "code_path": meta["code_path"],
            "image_path": img_path,
        })

    if len(instructions_seen) > 1:
        print(f"[WARN] Multiple instructions found within one episode block: {sorted(list(instructions_seen))[:3]}")

    if instruction is None:
        instruction = "unknown task"

    return instruction, candidates

# ---------------------------------------------------------------------
# Run VLM for one episode
# ---------------------------------------------------------------------

def filter_one_episode_with_vlm(
    client: OpenAI,
    instruction: str,
    candidates: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    max_side: int = 448,
    detail: str = "low",
) -> Dict[str, Any]:
    """
    VLM returns kept csv row ids only.
    """
    parts: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Task instruction: {instruction}"},
        {"type": "text", "text": "Below is the ordered list of candidate keysteps. Use the images as the primary source of truth. Apply only conservative filtering, and return the CSV row ids that should be kept."},
    ]

    for i, cand in enumerate(candidates):
        header = (
            f"{i:02d}. "
            f"csv_row_id={cand['csv_row_id']}, "
            f"step_keystep={cand['step_keystep']}, "
            f"keystep_type={cand['keystep_type']}"
        )
        parts.append({"type": "text", "text": header})

        data_url = _load_and_downscale_image_as_data_url(cand["image_path"], max_side=max_side)
        parts.append({
            "type": "image_url",
            "image_url": {"url": data_url, "detail": detail},
        })

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_FILTER_IMG}]},
            {"role": "user", "content": parts},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": FILTER_SCHEMA,
        },
    )

    data = json.loads(resp.choices[0].message.content)
    return data

# ---------------------------------------------------------------------
# Main filtering pipeline
# ---------------------------------------------------------------------

def run_filter_keystep_csv_with_vlm(
    manifest_csv: str,
    dataset_pkl: str,
    raw_image_root: str,
    out_csv: str,
    out_debug_dir: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_side: int = 448,
    api_key: Optional[str] = None,
    detail: str = "low",
    map_debug_print_limit: int = 10,
) -> pd.DataFrame:
    """
    Read original manifest CSV, filter row ids episode by episode, and save a new CSV
    with exactly the same schema as the original one.
    """
    client = _make_client(api_key)
    frame_map = build_global_frame_index_from_pkl(
        dataset_pkl,
        raw_image_root,
        debug_print_limit=map_debug_print_limit,
    )

    df = pd.read_csv(manifest_csv)
    original_columns = list(df.columns)

    df = df.copy().reset_index(drop=True)
    df["csv_row_id"] = df.index.astype(int)

    kept_row_ids: List[int] = []
    debug_rows: List[Dict[str, Any]] = []

    grouped = df.groupby("episode_id", sort=False)
    #grouped = list(df.groupby("episode_id", sort=False))[:10]
    for episode_id, df_ep in tqdm(grouped, desc="Filtering episodes"):
        instruction, candidates = build_episode_candidates(df_ep, frame_map)

        print(f"[INFO][EP {episode_id}] num_candidates={len(candidates)} | instruction={instruction[:80]}")

        result = filter_one_episode_with_vlm(
            client=client,
            instruction=instruction,
            candidates=candidates,
            model=model,
            max_side=max_side,
            detail=detail,
        )

        kept = result.get("kept_rows", [])
        kept = [int(x) for x in kept]

        valid_rows = set(int(x) for x in df_ep["csv_row_id"].tolist())
        kept = [x for x in kept if x in valid_rows]

        dropped = sorted(valid_rows - set(kept))

        print(f"[INFO][EP {episode_id}] kept={len(kept)} | dropped={len(dropped)} | filtered_ok={bool(result.get('filtered_ok', False))}")
        if result.get("problems", []):
            print(f"[INFO][EP {episode_id}] problems={result.get('problems', [])}")

        kept_row_ids.extend(kept)

        if out_debug_dir is not None:
            os.makedirs(out_debug_dir, exist_ok=True)
            debug_path = os.path.join(out_debug_dir, f"episode_{episode_id}.json")
            debug_data = {
                "episode_id": str(episode_id),
                "instruction": instruction,
                "candidates": [
                    {
                        "csv_row_id": c["csv_row_id"],
                        "idx_keystep": c["idx_keystep"],
                        "step_keystep": c["step_keystep"],
                        "keystep_type": c["keystep_type"],
                        "code_path": c["code_path"],
                        "raw_image_path": c["image_path"],
                    }
                    for c in candidates
                ],
                "vlm_result": result,
                "kept_rows": kept,
                "dropped_rows": dropped,
            }
            _write_json(debug_path, debug_data)

        debug_rows.append({
            "episode_id": str(episode_id),
            "num_candidates": len(candidates),
            "num_kept": len(kept),
            "num_dropped": len(dropped),
            "filtered_ok": bool(result.get("filtered_ok", False)),
            "problems": result.get("problems", []),
        })

    kept_row_ids = sorted(set(kept_row_ids))

    df_filtered = df[df["csv_row_id"].isin(kept_row_ids)].copy()
    df_filtered = df_filtered.sort_values(["episode_id", "step_keystep", "idx_keystep"])
    df_filtered = df_filtered[original_columns]

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_filtered.to_csv(out_csv, index=False)

    print(f"[INFO] Input rows:  {len(df)}")
    print(f"[INFO] Output rows: {len(df_filtered)}")
    print(f"[INFO] Filtered CSV saved to: {out_csv}")

    if out_debug_dir is not None:
        summary_path = os.path.join(out_debug_dir, "filter_summary.json")
        _write_json(summary_path, {
            "manifest_csv": manifest_csv,
            "dataset_pkl": dataset_pkl,
            "raw_image_root": raw_image_root,
            "out_csv": out_csv,
            "num_input_rows": int(len(df)),
            "num_output_rows": int(len(df_filtered)),
            "episode_summaries": debug_rows,
        })

    return df_filtered

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--dataset_pkl", type=str, required=True)
    parser.add_argument("--raw_image_root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_debug_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--max_side", type=int, default=448)
    parser.add_argument("--detail", type=str, default="low")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--map_debug_print_limit", type=int, default=10)
    args = parser.parse_args()

    run_filter_keystep_csv_with_vlm(
        manifest_csv=args.manifest_csv,
        dataset_pkl=args.dataset_pkl,
        raw_image_root=args.raw_image_root,
        out_csv=args.out_csv,
        out_debug_dir=args.out_debug_dir,
        model=args.model,
        max_side=args.max_side,
        api_key=args.api_key,
        detail=args.detail,
        map_debug_print_limit=args.map_debug_print_limit,
    )

if __name__ == "__main__":
    main()
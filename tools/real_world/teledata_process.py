# -*- coding: utf-8 -*-
"""
Process teleoperated real-robot data (5Hz CSV episodes) into structvla/Libero-like format:
- Traverse RAW_ROOT/<timestamp>/episodes_5hz/*.csv
- Downsample & save images into real_all_raw_{S}
- Save per-step actions into real_all/<scene>/actions/*.npy
- Save instruction into real_all/<scene>/instruction.txt
- (Optional) Run Emu3-VisionTokenizer to produce codes for both views
- Build normalized pickle: meta/real_all_norm.pkl
"""

import os
import os.path as osp
import re
import sys
import json
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image

import torch
from transformers import AutoModel, AutoImageProcessor

# ---- structvla normalization utils (same as your libero script) ----
PROJECT_ROOT = "/remote-home/jinminghao"
sys.path.append(f"{PROJECT_ROOT}/structvla")
from train.dataset.normalize_pi0 import RunningStats, save  # noqa: E402


# ---------------------------
# Helpers
# ---------------------------
def rewrite_instruction_to_move_to(instr: str) -> str:
    """
    Rewrite instruction to: 'move to <last object phrase>.'
    Heuristic: take substring after the last occurrence of common prepositions
    (in/into/on/onto/to) + 'the/a/an', else fall back to last 'the ...' phrase,
    else use the last 2-4 words.
    """
    if instr is None:
        return "move to the object."
    s = str(instr).strip().lower()

    # strip trailing punctuation
    s = re.sub(r"[\.!\?]+$", "", s).strip()

    # common patterns where the last object appears after a preposition
    patterns = [
        r"\binto\s+(the|a|an)\s+(.+)$",
        r"\bin\s+(the|a|an)\s+(.+)$",
        r"\bonto\s+(the|a|an)\s+(.+)$",
        r"\bon\s+(the|a|an)\s+(.+)$",
        r"\bto\s+(the|a|an)\s+(.+)$",
    ]

    for pat in patterns:
        m = re.search(pat, s)
        if m:
            obj = m.group(2).strip()
            obj = re.sub(r"\s+", " ", obj)
            # ensure it starts with 'the' for consistency
            return f"move to the {obj}."

    # fallback: take the last "the <...>" chunk if exists
    m2 = re.findall(r"\bthe\s+([a-z0-9_\- ]+)", s)
    if m2:
        obj = m2[-1].strip()
        obj = re.sub(r"\s+", " ", obj)
        return f"move to the {obj}."

    # last resort: take last 3 words
    toks = re.sub(r"[^a-z0-9 ]+", " ", s).split()
    tail = " ".join(toks[-3:]) if len(toks) >= 3 else " ".join(toks)
    tail = tail.strip() if tail else "object"
    return f"move to the {tail}."

def parse_crop_box(s: str):
    """
    Parse "left,top,right,bottom" -> (l,t,r,b) as int tuple.
    Return None if s is empty/None.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid crop_box '{s}', expect 'left,top,right,bottom'")
    l, t, r, b = parts
    if r <= l or b <= t:
        raise ValueError(f"Invalid crop_box '{s}', right/bottom must be > left/top")
    return (l, t, r, b)

def crop_then_resize(im: Image.Image, crop_box, size_hw):
    """
    im: PIL.Image RGB
    crop_box: (l,t,r,b) in pixel coords of original image, or None for no crop
    size_hw: (W,H) for final resize
    """
    if crop_box is not None:
        W0, H0 = im.size
        l, t, r, b = crop_box
        # clamp for safety
        l = max(0, min(l, W0))
        r = max(0, min(r, W0))
        t = max(0, min(t, H0))
        b = max(0, min(b, H0))
        if r > l and b > t:
            im = im.crop((l, t, r, b))
    return im.resize(size_hw, resample=Image.BICUBIC)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def sanitize_name(x: str) -> str:
    # Replace ':' and spaces etc. to make safe folder names
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", x)

def resolve_path(rel_or_abs: str, candidate_roots):
    """
    Resolve a path stored in CSV. Try:
      1) as-is (absolute)
      2) join with each candidate root
    """
    if rel_or_abs is None:
        return None
    p = str(rel_or_abs)
    if osp.isabs(p) and osp.exists(p):
        return p
    for r in candidate_roots:
        cand = osp.normpath(osp.join(r, p))
        if osp.exists(cand):
            return cand
    return None

def load_rgb(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im

def sort_frame_files(folder: str, ext=".npy"):
    files = [osp.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)]
    files = sorted(files, key=lambda x: int(osp.splitext(osp.basename(x))[0]))
    return files

@torch.no_grad()
def encode_folder_to_codes(
    image_dir: str,
    out_code_dir: str,
    model,
    processor,
    size_hw,
    batch_size: int = 8,
):
    ensure_dir(out_code_dir)
    img_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=natural_sort_key
    )
    if not img_files:
        return 0

    paths = [osp.join(image_dir, f) for f in img_files]
    total = 0

    for st in range(0, len(paths), batch_size):
        batch_paths = paths[st:st + batch_size]
        images = [load_rgb(p).resize(size_hw, resample=Image.BICUBIC) for p in batch_paths]
        pixel_values = processor(images, return_tensors="pt")["pixel_values"].cuda(non_blocking=True)

        codes = model.encode(pixel_values)              # torch tensor, shape usually (B,H,W) or (B,1,H,W)
        codes = codes.detach().cpu().numpy()

        for i, p in enumerate(batch_paths):
            frame_id = osp.splitext(osp.basename(p))[0]  # "000000"
            c = codes[i]

            # Key point: always save as (1, H, W)
            if c.ndim == 2:           # (H,W)
                c = c[None, ...]      # (1,H,W)
            elif c.ndim == 3:
                # Allow (1, H, W); if shape is (C, H, W) with C != 1, fail fast because the current pipeline does not support it
                if c.shape[0] != 1:
                    raise RuntimeError(f"Unexpected code shape {c.shape} for {p}, expected (H,W) or (1,H,W)")
            else:
                raise RuntimeError(f"Unexpected code ndim {c.ndim} shape {c.shape} for {p}")

            # Prefer int32 to save space; int64 also works but is larger
            np.save(osp.join(out_code_dir, f"{frame_id}.npy"), c.astype(np.int64))
            total += 1

    return total



# ---------------------------
# Main pipeline
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, nargs="+", required=True,
                    help="one or more raw roots, e.g. /path/a /path/b")
    ap.add_argument("--out_root", type=str, required=True,
                    help="e.g. /remote-home/share/teledata/raw/2025-12-23-split1")

    # Image downsample

    ap.add_argument("--jpeg_quality", type=int, default=95)

    # Episode filtering
    ap.add_argument("--min_frames", type=int, default=8)
    ap.add_argument("--drop_last", action="store_true",
                    help="Drop last step to make (obs_t, action_t)->obs_{t+1} alignment safer")

    # Action/gripper
    ap.add_argument("--gripper_mode", type=str, default="absolute_next",
                    choices=["absolute_curr", "absolute_next", "toggle"])
    ap.add_argument("--include_gripper_in_action", action="store_true",
                    help="If set: action dim = 7 (6DoF + gripper). Otherwise dim = 6.")

    # Tokenizer
    ap.add_argument("--run_encode", action="store_true",
                    help="Run Emu3-VisionTokenizer to produce codes")
    ap.add_argument("--vision_tokenizer_path", type=str,
                    default="/remote-home/jinminghao/structvla/pretrain/Emu3-VisionTokenizer")
    ap.add_argument("--vision_tokenizer_hub", type=str,
                    default="BAAI/Emu3-VisionTokenizer")
    ap.add_argument("--min_pixels", type=int, default=128 * 128)
    ap.add_argument("--encode_batch_size", type=int, default=8)

    # Build pickle + normalizer
    ap.add_argument("--build_pkl", action="store_true",
                    help="Build normalized pickle meta/real_all_norm.pkl")
    ap.add_argument("--normalizer_out", type=str, default="configs/normalizer_real_all")
    ap.add_argument("--pkl_out", type=str, default="meta/real_all_norm.pkl")
    ap.add_argument("--distributed_encode", action="store_true",
                    help="Use torchrun multi-GPU to encode scenes in parallel (split by rank)")
        # Image resize: non-square (W,H). Keep --size for backward compatibility.
    ap.add_argument("--size_hw", type=int, nargs=2, default=[256, 144],
                    help="Resize to (W H), e.g. --size_hw 256 144")
    ap.add_argument("--size_hw_wrist", type=int, nargs=2, default=[240, 135],
                    help="Resize wrist to (W H). If None, use --size_hw.")

    ap.add_argument("--size", type=int, default=None,
                    help="[Deprecated] If set, override size_hw and use square resize (size,size)")

    # Movement-only: drop frames after first gripper state change
    ap.add_argument("--keep_before_gripper_change", action="store_true",
                    help="Keep only frames before the first gripper state change (movement-only)")
    ap.add_argument("--gripper_threshold", type=float, default=0.5,
                    help="Threshold to binarize gripper state when detecting change")
    ap.add_argument("--rewrite_instruction_move_to", action="store_true",
                    help="Rewrite instruction into 'move to <last object>.' (movement-only dataset)")
    ap.add_argument("--crop_main", type=str, default="",
                help="Crop box for main image: 'left,top,right,bottom,300,20,900,620' before resize. Empty=no crop.")
    ap.add_argument("--crop_wrist", type=str, default="",
                    help="Crop box for wrist image: 'left,top,right,bottom' before resize. Empty=no crop.")
    ap.add_argument("--use_wrist_image", type=bool, default=True,
                    help="True: process wrist/gripper images; False: ignore wrist images completely")

    
    args = ap.parse_args()
    IMG_COL = "Front_Image"   # Read the front view, still save to images/, and keep the PKL key as image

    # resolve final resize size
    # main size
    if args.size is not None:
        Wm = Hm = int(args.size)
    else:
        Wm, Hm = int(args.size_hw[0]), int(args.size_hw[1])
    SIZE_MAIN = (Wm, Hm)

    # wrist size
    if args.size is not None:
        Ww = Hw = int(args.size)   # If --size is set, force both streams to use the same size for compatibility
    else:
        if args.size_hw_wrist is None:
            Ww, Hw = Wm, Hm        # Default to the same size as the main stream
        else:
            Ww, Hw = int(args.size_hw_wrist[0]), int(args.size_hw_wrist[1])
    SIZE_WRIST = (Ww, Hw)

    crop_main = parse_crop_box(args.crop_main)
    crop_wrist = parse_crop_box(args.crop_wrist)


    # Output dirs
    language_dir = osp.join(args.out_root, "real_all")
    raw_img_dir = osp.join(args.out_root, f"real_all_raw_main_{Wm}x{Hm}_wrist_{Ww}x{Hw}")
    code_dir    = osp.join(args.out_root, f"real_all_codes_main_{Wm}x{Hm}")
    wrist_code_dir = osp.join(args.out_root, f"real_all_gripper_codes_wrist_{Ww}x{Hw}")

    ensure_dir(language_dir)
    ensure_dir(raw_img_dir)
    ensure_dir(code_dir)
    ensure_dir(wrist_code_dir)
    ensure_dir(osp.join(args.out_root, "meta"))
    ensure_dir(osp.join(args.out_root, "configs"))

    # 1) Traverse and prepare episodes
    # RAW_ROOT/<timestamp>/episodes_5hz/*.csv


    all_scenes = []

    for raw_root in args.raw_root:
        timestamp_dirs = sorted(
            [d for d in glob(osp.join(raw_root, "*")) if osp.isdir(d)],
            key=natural_sort_key
        )

        root_name = sanitize_name(osp.basename(raw_root))

        for ts_dir in tqdm(timestamp_dirs, desc=f"Scanning timestamps in {root_name}"):
            epi_dir = osp.join(ts_dir, "episodes_5hz")
            if not osp.isdir(epi_dir):
                continue
            csv_files = sorted(glob(osp.join(epi_dir, "*.csv")), key=natural_sort_key)
            if not csv_files:
                continue

            ts_name = sanitize_name(osp.basename(ts_dir))

            for csv_path in csv_files:
                csv_base = sanitize_name(osp.splitext(osp.basename(csv_path))[0])
                scene = f"{root_name}__{ts_name}__{csv_base}"
                scene_lang = osp.join(language_dir, scene)
                scene_raw = osp.join(raw_img_dir, scene)

                instr_path = osp.join(scene_lang, "instruction.txt")
                actions_path = osp.join(scene_lang, "actions")
                img_out_dir = osp.join(scene_raw, "images")
                wrist_out_dir = osp.join(scene_raw, "wrist_images")

                # Skip if already prepared
                ok_basic = osp.exists(instr_path) and osp.isdir(actions_path) and osp.isdir(img_out_dir)
                ok_wrist = (osp.isdir(wrist_out_dir) if args.use_wrist_image else True)
                if ok_basic and ok_wrist:
                    all_scenes.append(scene)
                    continue

                ensure_dir(scene_lang)
                ensure_dir(actions_path)
                ensure_dir(img_out_dir)
                if args.use_wrist_image:
                    ensure_dir(wrist_out_dir)
                df = pd.read_csv(csv_path)

                # Sort by Time if available
                if "Time" in df.columns:
                    df = df.sort_values("Time").reset_index(drop=True)

                # Basic required columns
                required = [IMG_COL, "task",
                            "action_x","action_y","action_z","action_roll","action_pitch","action_yaw"]
                if args.use_wrist_image:
                    required.insert(1, "Wrist_Image")
                for c in required:
                    if c not in df.columns:
                        raise KeyError(f"[{csv_path}] missing column: {c}")

                # Candidate roots for resolving relative paths
                # 1) directory of CSV
                # 2) timestamp dir
                # 3) episodes_5hz dir
                candidate_roots = [osp.dirname(csv_path), ts_dir, epi_dir, raw_root]


                # Drop rows with missing
                df = df.dropna(subset=[IMG_COL,"Wrist_Image","task"]).reset_index(drop=True)
                if len(df) < args.min_frames:
                    continue

                # Instruction
                instruction_raw = str(df.loc[0, "task"])
                instruction = rewrite_instruction_to_move_to(instruction_raw) if args.rewrite_instruction_move_to else instruction_raw
                with open(instr_path, "w") as f:
                    f.write(instruction)

                # Gripper state (0 open, 1 close) optional
                if "Gripper" in df.columns:
                    grippers = df["Gripper"].astype(float).to_numpy()
                else:
                    grippers = np.zeros(len(df), dtype=np.float32)

                # Build actions array (T,6) then optional append gripper action
                a6 = df[["action_x","action_y","action_z","action_roll","action_pitch","action_yaw"]].astype(float).to_numpy()

                if args.gripper_mode == "absolute_curr":
                    g_act = grippers
                elif args.gripper_mode == "absolute_next":
                    g_act = np.concatenate([grippers[1:], grippers[-1:]], axis=0)
                else:  # toggle
                    g_act = np.concatenate([(grippers[1:] != grippers[:-1]).astype(np.float32), [0.0]], axis=0)

                if args.include_gripper_in_action:
                    actions = np.concatenate([a6, g_act[:, None]], axis=1).astype(np.float32)
                else:
                    actions = a6.astype(np.float32)
                # -----------------------------
                # [NEW] movement-only trimming:
                # keep frames strictly BEFORE the first gripper state change
                # -----------------------------
                if args.keep_before_gripper_change:
                    if "Gripper" not in df.columns:
                        print(f"[Warn] {csv_path}: keep_before_gripper_change is on but no 'Gripper' column. Skip trimming.")
                    else:
                        g_bin = (grippers > args.gripper_threshold).astype(np.int8)
                        # diff index i means change from i -> i+1, so changed frame index is i+1
                        change_pos = np.nonzero(g_bin[1:] != g_bin[:-1])[0]
                        if len(change_pos) > 0:
                            cut = int(change_pos[0] + 1)  # keep [0, cut)
                            df = df.iloc[:cut].reset_index(drop=True)
                            actions = actions[:cut]
                            grippers = grippers[:cut]
                # Optional drop last
                if args.drop_last:
                    df = df.iloc[:-1].reset_index(drop=True)
                    actions = actions[:-1]
                    grippers = grippers[:-1]

                if len(df) < args.min_frames:
                    continue

                # Save downsampled images and per-step action npys
                for t in range(len(df)):
                    # Resolve input image paths
                    img_in = resolve_path(df.loc[t, IMG_COL], candidate_roots)

                    if img_in is None:
                        raise FileNotFoundError(
                            f"Cannot resolve image paths in {csv_path} at row {t}: "
                            f"{IMG_COL}={df.loc[t, IMG_COL]}"
                        )

                    im = crop_then_resize(load_rgb(img_in), crop_main, SIZE_MAIN)

                    im.save(osp.join(img_out_dir, f"{t:06d}.jpg"), quality=args.jpeg_quality)


                    if args.use_wrist_image:
                        wrist_in = resolve_path(df.loc[t, "Wrist_Image"], candidate_roots)
                        if wrist_in is None:
                            raise FileNotFoundError(
                                f"Cannot resolve image paths in {csv_path} at row {t}: "
                                f"Wrist_Image={df.loc[t,'Wrist_Image']}"
                            )
                        wm = crop_then_resize(load_rgb(wrist_in), crop_wrist, SIZE_WRIST)
                        wm.save(osp.join(wrist_out_dir, f"{t:06d}.jpg"), quality=args.jpeg_quality)

                    np.save(osp.join(actions_path, f"{t:06d}.npy"), actions[t])

                all_scenes.append(scene)

    all_scenes = sorted(list(set(all_scenes)))
    print(f"Prepared scenes: {len(all_scenes)}")
    if not all_scenes:
        raise RuntimeError("No valid scenes prepared. Check raw_root structure and CSV contents.")

    # 2) Optional: encode tokens
    if args.run_encode:
        # --- DDP info from torchrun ---
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        rank = int(os.getenv("RANK", str(local_rank)))

        if args.distributed_encode and world_size > 1:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)

            # Initialize the process group for barriers; it can run without it, but barriers are safer
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")

            # Split scenes so each rank handles a subset
            scenes_this = all_scenes[rank::world_size]
        else:
            device = torch.device("cuda", 0)
            scenes_this = all_scenes

        print(f"[encode] rank={rank} local_rank={local_rank} world_size={world_size} scenes={len(scenes_this)}")

        print("Loading Emu3-VisionTokenizer...")
        model = AutoModel.from_pretrained(args.vision_tokenizer_path, trust_remote_code=True).eval().to(device)
        processor = AutoImageProcessor.from_pretrained(args.vision_tokenizer_hub, trust_remote_code=True)
        processor.min_pixels = args.min_pixels

        for scene in tqdm(scenes_this, desc=f"Encoding codes (rank {rank})"):
            scene_raw = osp.join(raw_img_dir, scene)
            main_in = osp.join(scene_raw, "images")
            wrist_in = osp.join(scene_raw, "wrist_images")

            main_out = osp.join(code_dir, scene)
            wrist_out = osp.join(wrist_code_dir, scene)

            # Skip if already encoded
            ok_main = osp.isdir(main_out) and len(os.listdir(main_out)) > 0
            ok_wrist = (osp.isdir(wrist_out) and len(os.listdir(wrist_out)) > 0) if args.use_wrist_image else True
            if ok_main and ok_wrist:
                continue

            encode_folder_to_codes(
                image_dir=main_in,
                out_code_dir=main_out,
                model=model,
                processor=processor,
                size_hw=SIZE_MAIN,
                batch_size=args.encode_batch_size,
            )
            if args.use_wrist_image:
                encode_folder_to_codes(
                    image_dir=wrist_in,
                    out_code_dir=wrist_out,
                    model=model,
                    processor=processor,
                    size_hw=SIZE_WRIST,
                    batch_size=args.encode_batch_size,
                )
        # Wait for all ranks to finish encoding before moving to later steps such as build_pkl
        if args.distributed_encode and world_size > 1:
            torch.distributed.barrier()

    # 3) Optional: build normalized pickle (Libero-style dict keys)
    if args.build_pkl:
        result_file = []
        min_frames = args.min_frames

        for scene in tqdm(all_scenes, desc="Building pickle"):
            instr_file = osp.join(language_dir, scene, "instruction.txt")
            action_folder = osp.join(language_dir, scene, "actions")
            if not osp.exists(instr_file) or not osp.isdir(action_folder):
                continue

            with open(instr_file, "r") as f:
                text_raw = f.read()
            text = rewrite_instruction_to_move_to(text_raw) if args.rewrite_instruction_move_to else text_raw

            action_files = sorted(
                [osp.join(action_folder, f) for f in os.listdir(action_folder) if f.endswith(".npy")],
                key=lambda x: int(osp.splitext(osp.basename(x))[0])
            )
            if len(action_files) < min_frames:
                continue
            action = [np.load(a) for a in action_files]

            # Need codes
            img_dir = osp.join(code_dir, scene)
            if not osp.isdir(img_dir):
                continue

            img_files = sort_frame_files(img_dir, ext=".npy")

            if args.use_wrist_image:
                grip_dir = osp.join(wrist_code_dir, scene)
                if not osp.isdir(grip_dir):
                    continue
                grip_files = sort_frame_files(grip_dir, ext=".npy")
                L = min(len(action), len(img_files), len(grip_files))
            else:
                grip_files = None
                L = min(len(action), len(img_files))

            if L < min_frames:
                continue
            action = action[:L]
            img_files = img_files[:L]
            if args.use_wrist_image:
                grip_files = grip_files[:L]
            # -----------------------------
            # [NEW] movement-only trimming at pkl stage
            # (works only if include_gripper_in_action=True, i.e. action dim == 7)
            # -----------------------------
            if args.keep_before_gripper_change:
                a_stack = np.stack(action, axis=0)  # (T, A)
                if a_stack.shape[1] >= 7:
                    g = a_stack[:, -1]
                    g_bin = (g > args.gripper_threshold).astype(np.int8)
                    change_pos = np.nonzero(g_bin[1:] != g_bin[:-1])[0]
                    if len(change_pos) > 0:
                        cut = int(change_pos[0] + 1)
                        action = action[:cut]
                        img_files = img_files[:cut]
                        if args.use_wrist_image:
                            grip_files = grip_files[:cut]
                else:
                    print(f"[Warn] scene={scene}: keep_before_gripper_change is on but action_dim={a_stack.shape[1]} < 7; cannot trim in build_pkl.")

            result_file.append({
                "text": text,
                "image": img_files,
                "action": action,
                "gripper_image": grip_files,
            })

        print(f"Total valid scenes for pkl: {len(result_file)}")
        if not result_file:
            raise RuntimeError("No valid scenes for pickle. If you didn't run encoding, add --run_encode.")

        # Normalize actions to [-1, 1] using Q01/Q99 (same as your libero script)
        normalizer = RunningStats()
        action_data = np.concatenate([np.stack(scene["action"], axis=0) for scene in result_file], axis=0)
        normalizer.update(action_data)
        stats = normalizer.get_statistics()

        print("Mean:", stats.mean)
        print("Std:", stats.std)
        print("Q01:", stats.q01)
        print("Q99:", stats.q99)

        for scene in result_file:
            a = np.stack(scene["action"], axis=0)
            normalized = 2 * (a - stats.q01) / (stats.q99 - stats.q01 + 1e-8) - 1
            scene["action"] = np.clip(normalized, -1, 1)

        # Save pkl
        pkl_out = osp.join(args.out_root, args.pkl_out)
        ensure_dir(osp.dirname(pkl_out))
        import pickle
        with open(pkl_out, "wb") as f:
            pickle.dump(result_file, f, protocol=4)
        print(f"Saved normalized pkl to: {pkl_out}")

        # Save normalizer stats
        normalizer_path = osp.join(args.out_root, args.normalizer_out)
        ensure_dir(normalizer_path)
        save(normalizer_path, {"real_all": stats})
        print(f"Saved normalizer statistics to: {normalizer_path}")

    print("Done.")


if __name__ == "__main__":
    main()

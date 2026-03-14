#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keysteps extractor - Final Version
1. Output: save keysteps.csv and summary.json under out_dir
2. Logic: Start (removed) -> [Still/Gap] -> Flip -> End
3. Parameters: Still thresholds are slightly tuned to keep a moderate number of still frames; Gap Fill keeps the >15 limit
"""

import argparse, os, sys, json, pickle
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------ Utils ------------------------

def log(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}")

def find_change_points(bits: List[int]) -> List[int]:
    return [i for i in range(1, len(bits)) if bits[i] != bits[i-1]]

def ema_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x.astype(np.float32)
    alpha = 2.0 / (win + 1.0)
    y = np.empty_like(x, dtype=np.float32)
    acc = float(x[0])
    for i in range(len(x)):
        acc = alpha * float(x[i]) + (1.0 - alpha) * acc
        y[i] = acc
    return y

def action_inst_mags(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if actions.ndim != 2 or actions.shape[1] < 6:
        T = actions.shape[0]
        return np.full(T, 1e9, dtype=np.float32), np.full(T, 1e9, dtype=np.float32)
    
    pos_mag = np.sqrt(np.sum(np.square(actions[:, :3].astype(np.float32)), axis=1))
    rot_mag = np.sqrt(np.sum(np.square(actions[:, 3:6].astype(np.float32)), axis=1))
    return pos_mag.astype(np.float32), rot_mag.astype(np.float32)

def binarize_with_mode(x: float, mode: str = "gt0") -> int:
    if mode == "gt05": return int(x > 0.5)
    elif mode == "gt0": return int(x > 0.0)
    else: return int(np.sign(x) > 0)

# ------------------------ Data Loading ------------------------

def flatten_bridge_pkl(pkl_path: str, test: bool = False) -> List[Dict[str, Any]]:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")
        
    with open(pkl_path, "rb") as f:
        episodes = pickle.load(f)
    
    if not isinstance(episodes, list):
        if isinstance(episodes, dict): episodes = [episodes]
        else: raise ValueError("PKL content must be List or Dict")
        
    if test: episodes = episodes[:5]

    rows = []
    for epi, ep in enumerate(episodes):
        if "action" not in ep: continue
        actions = np.asarray(ep["action"])
        if actions.ndim != 2 or actions.shape[1] < 6: continue
            
        T = actions.shape[0]
        imgs = ep.get("image", [""] * T)
        if len(imgs) < T: imgs = imgs + [""] * (T - len(imgs))
        
        ep_name = (
            ep.get("episode_name") or ep.get("task_name") or 
            ep.get("task") or ep.get("name") or f"ep_{epi}"
        )
        for t in range(T):
            rows.append({
                "episode_id": str(epi),
                "ep_name": ep_name,
                "step": t,
                "action": actions[t],
                "gripper": float(actions[t, -1]),
                "image": imgs[t],
            })
    return rows

# ------------------------ Thresholds & Logic ------------------------

def compute_episode_thresholds(actions: np.ndarray,
                               flip_pos_pctl: float, flip_rot_pctl: float,
                               still_pos_pctl: float, still_rot_pctl: float,
                               flip_scale: float, still_scale: float,
                               flip_pos_min: float, flip_rot_min: float,
                               still_pos_max: float, still_rot_max: float) -> Dict[str, float]:
    pos_mag, rot_mag = action_inst_mags(actions)
    pos_mag = np.nan_to_num(pos_mag, 0.0)
    rot_mag = np.nan_to_num(rot_mag, 0.0)

    flip_pos = max(np.percentile(pos_mag, flip_pos_pctl) * flip_scale, flip_pos_min)
    flip_rot = max(np.percentile(rot_mag, flip_rot_pctl) * flip_scale, flip_rot_min)

    still_pos = min(np.percentile(pos_mag, still_pos_pctl) * still_scale, still_pos_max)
    still_rot = min(np.percentile(rot_mag, still_rot_pctl) * still_scale, still_rot_max)

    return {
        "flip_enter_pos": float(flip_pos),
        "flip_enter_rot": float(flip_rot),
        "still_pos": float(still_pos),
        "still_rot": float(still_rot),
    }

def find_grip_settle_end(g_values: np.ndarray, grip_bits: List[int], cp: int,
                         new_state: int, settle_eps: float, settle_win: int, T: int) -> int:
    win = max(1, int(settle_win))
    if cp >= T: return T - 1
    dg = np.zeros(T, dtype=np.float32)
    dg[1:] = np.abs(g_values[1:] - g_values[:-1])
    for t in range(cp, T - win + 1):
        ok = True
        for u in range(t, t + win):
            if grip_bits[u] != new_state or dg[u] > settle_eps:
                ok = False; break
        if ok: return t + win - 1
    return cp

def pick_keystep_after_grip_settle(frames: List[Dict[str, Any]],
                                   indices: List[int],
                                   cp: int,
                                   grip_bits: List[int],
                                   g_values: np.ndarray,
                                   enter_pos: float, enter_rot: float,
                                   hysteresis: float,
                                   ema_win: int,
                                   max_lookahead: int,
                                   min_offset_after_flip: int,
                                   grip_settle_eps: float,
                                   grip_settle_win: int,
                                   pre_move_horizon: int) -> Dict[str, Any]:
    new_state = grip_bits[cp]
    T = len(indices)
    acts = np.stack([np.asarray(frames[int(i)]["action"], dtype=np.float32) for i in indices], axis=0)
    pos_mag, rot_mag = action_inst_mags(acts)
    
    pos_s = ema_1d(pos_mag, ema_win)
    rot_s = ema_1d(rot_mag, ema_win)

    settle_end = find_grip_settle_end(g_values, grip_bits, cp, new_state,
                                      settle_eps=grip_settle_eps, settle_win=grip_settle_win, T=T)
    
    search_start = min(max(cp + max(1, int(min_offset_after_flip)), settle_end), T-1)
    end_t = min(cp + int(max_lookahead), T - 1)

    exit_pos_thr  = enter_pos * float(hysteresis)
    exit_rot_thr  = enter_rot * float(hysteresis)

    last_still_t = search_start
    found_premove = False
    
    # Default outputs
    ks_t = last_still_t

    t = search_start
    while t <= end_t:
        if grip_bits[t] != new_state: break
        
        horizon_end = min(t + int(pre_move_horizon), T - 1)
        first_big_k = None
        for k in range(t, horizon_end + 1):
            if (pos_s[k] > enter_pos) or (rot_s[k] > enter_rot):
                first_big_k = k
                break

        if first_big_k is not None:
            ks_t = max(settle_end, first_big_k - 1)
            found_premove = True
            break 

        if (pos_s[t] <= exit_pos_thr) and (rot_s[t] <= exit_rot_thr):
            last_still_t = t
        t += 1
    
    if not found_premove:
        ks_t = int(last_still_t)

    idx_global = int(indices[ks_t])
    idx_before = int(indices[max(0, ks_t - 1)])
    
    return {
        "type": "flip",
        "idx_before": idx_before,
        "idx_keystep": idx_global,
        "step_before": int(frames[idx_before]["step"]),
        "step_keystep": int(frames[idx_global]["step"]),
        "gripper_before": int(grip_bits[cp-1]),
        "gripper_after": int(grip_bits[cp]),
        "pos_delta_keystep": float(pos_mag[ks_t]),
        "rot_delta_keystep": float(rot_mag[ks_t]),
        "found_premove": bool(found_premove),
        "lookahead_used": 0,
        "binarize_mode": "gt0"
    }

def collect_still_keysteps(frames, indices, steps, actions, grip_bits,
                           pos_mag_thresh, rot_mag_thresh, win, min_gap,
                           existing_steps, flip_steps, flip_guard, max_scale,
                           ema_win, hysteresis, ratio_req) -> List[Dict[str, Any]]:
    T = actions.shape[0]
    if T == 0: return []

    pos_mag, rot_mag = action_inst_mags(actions)
    pos_s = ema_1d(pos_mag, ema_win)
    rot_s = ema_1d(rot_mag, ema_win)

    pos_exit = float(pos_mag_thresh) * float(hysteresis)
    rot_exit = float(rot_mag_thresh) * float(hysteresis)

    used = set(int(s) for s in existing_steps)
    flip_set = set(int(s) for s in flip_steps)
    chosen = []
    last_still_step = None
    w = max(1, int(win))

    for t in range(T):
        if t < w - 1: continue
        step_t = int(steps[t])

        if any(abs(step_t - s) < min_gap for s in used): continue
        if (last_still_step is not None) and abs(step_t - last_still_step) < min_gap: continue
        if any(abs(step_t - fs) <= flip_guard for fs in flip_set): continue

        t0 = t - w + 1
        bit_ref = grip_bits[t0]
        if any(grip_bits[u] != bit_ref for u in range(t0, t + 1)): continue

        pos_raw_win = pos_mag[t0:t+1]
        rot_raw_win = rot_mag[t0:t+1]
        
        count_ok = np.sum((pos_raw_win <= pos_mag_thresh) & (rot_raw_win <= rot_mag_thresh))
        if float(count_ok) / float(w) < float(ratio_req): continue

        if np.max(pos_raw_win) > (pos_mag_thresh * max_scale): continue
        if np.max(rot_raw_win) > (rot_mag_thresh * max_scale): continue

        if (pos_s[t] > pos_exit) or (rot_s[t] > rot_exit): continue

        idx_global = int(indices[t])
        idx_before = int(indices[max(0, t-1)])
        
        chosen.append({
            "type": "still",
            "idx_before": idx_before,
            "idx_keystep": idx_global,
            "step_before": int(frames[idx_before]["step"]),
            "step_keystep": step_t,
            "gripper_before": int(grip_bits[t]),
            "gripper_after": int(grip_bits[t]),
            "pos_delta_keystep": float(pos_s[t]),
            "rot_delta_keystep": float(rot_s[t]),
            "found_premove": False,
            "lookahead_used": 0,
            "binarize_mode": "gt0"
        })
        used.add(step_t)
        last_still_step = step_t

    return chosen

def make_full_item(frames, indices, steps, local_idx, item_type="gap_fill") -> Dict[str, Any]:
    idx_g = int(indices[local_idx])
    local_before = max(0, local_idx - 1)
    idx_b = int(indices[local_before])
    
    act = np.array([frames[idx_g]["action"]], dtype=np.float32)
    pm, rm = action_inst_mags(act)
    
    g_curr = frames[idx_g]["gripper"]
    g_prev = frames[idx_b]["gripper"]
    
    return {
        "type": item_type,
        "idx_before": idx_b,
        "idx_keystep": idx_g,
        "step_before": int(frames[idx_b]["step"]),
        "step_keystep": int(frames[idx_g]["step"]),
        "gripper_before": binarize_with_mode(g_prev, "gt0"),
        "gripper_after": binarize_with_mode(g_curr, "gt0"),
        "pos_delta_keystep": float(pm[0]),
        "rot_delta_keystep": float(rm[0]),
        "found_premove": False,
        "lookahead_used": 0,
        "binarize_mode": "gt0"
    }

# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--test", action="store_true")
    
    # === Parameter tuning for Still frame selection ===
    # Relax Pctl and Max slightly to allow more Still frames
    ap.add_argument("--still_pos_pctl", type=float, default=20.0) # (was 20)
    ap.add_argument("--still_rot_pctl", type=float, default=25.0)
    ap.add_argument("--still_pos_max", type=float, default=0.35)  # (was 0.35)
    ap.add_argument("--still_rot_max", type=float, default=0.40)
    # Key change: reduce the interval from 25 to 10 so 30-40 frame episodes can keep 1-2 Still frames
    ap.add_argument("--still_min_gap", type=int, default=15) 
    
    # === Flip parameters ===
    ap.add_argument("--flip_pos_pctl", type=float, default=65.0)
    ap.add_argument("--flip_rot_pctl", type=float, default=65.0)
    ap.add_argument("--flip_pos_min", type=float, default=0.35)
    ap.add_argument("--flip_rot_min", type=float, default=0.35)
    
    # === Auxiliary parameters ===
    ap.add_argument("--flip_scale", type=float, default=1.0)
    ap.add_argument("--still_scale", type=float, default=1.5)
    ap.add_argument("--still_win", type=int, default=4)
    ap.add_argument("--still_flip_guard", type=int, default=2)
    ap.add_argument("--still_max_scale", type=float, default=2.0)
    ap.add_argument("--still_ema_win", type=int, default=3)
    ap.add_argument("--still_hysteresis", type=float, default=1.0)
    ap.add_argument("--still_ratio", type=float, default=0.6)
    
    ap.add_argument("--max_lookahead", type=int, default=15)
    ap.add_argument("--min_offset_after_flip", type=int, default=0)
    ap.add_argument("--pre_move_horizon", type=int, default=3)
    ap.add_argument("--grip_settle_eps", type=float, default=0.05)
    ap.add_argument("--grip_settle_win", type=int, default=3)
    ap.add_argument("--flip_ema_win", type=int, default=3)
    ap.add_argument("--flip_hysteresis", type=float, default=0.8)
    ap.add_argument("--no_still_keys", action="store_true")

    # === Sparse gap filling parameters ===
    ap.add_argument("--sparse_gap_limit", type=int, default=20, help="when gap_fill")
    
    args = ap.parse_args()

    # Create the output directory
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv_path = os.path.join(args.out_dir, "keysteps.csv")
    out_json_path = os.path.join(args.out_dir, "summary.json")

    log(f"Loading {args.dataset} ...")
    frames = flatten_bridge_pkl(args.dataset, test=args.test)
    
    grouped: Dict[str, List[int]] = {}
    for i, fr in enumerate(frames):
        grouped.setdefault(fr["episode_id"], []).append(i)
    
    all_rows = []
    
    log("Processing episodes...")
    for ep_id, indices in tqdm(grouped.items(), desc="Episodes"):
        if not indices: continue
        
        g_values = np.array([float(frames[i]["gripper"]) for i in indices], dtype=np.float32)
        steps    = [int(frames[i]["step"]) for i in indices]
        actions  = np.stack([np.asarray(frames[i]["action"], dtype=np.float32) for i in indices], axis=0)
        grip_bits = (g_values > 0.0).astype(int).tolist()
        cps = find_change_points(grip_bits)

        th = compute_episode_thresholds(
            actions,
            args.flip_pos_pctl, args.flip_rot_pctl,
            args.still_pos_pctl, args.still_rot_pctl,
            args.flip_scale, args.still_scale,
            args.flip_pos_min, args.flip_rot_min,
            args.still_pos_max, args.still_rot_max
        )

        # 1. Flip
        flip_keys: List[Dict] = []
        for cp in cps:
            res_dict = pick_keystep_after_grip_settle(
                frames, indices, cp, grip_bits, g_values,
                th["flip_enter_pos"], th["flip_enter_rot"],
                args.flip_hysteresis, args.flip_ema_win,
                args.max_lookahead, args.min_offset_after_flip,
                args.grip_settle_eps, args.grip_settle_win, args.pre_move_horizon
            )
            flip_keys.append(res_dict)

        # 2. Still
        st_keys: List[Dict] = []
        if not args.no_still_keys:
            st_keys = collect_still_keysteps(
                frames, indices, steps, actions, grip_bits,
                th["still_pos"], th["still_rot"],
                args.still_win, args.still_min_gap,
                [k["step_keystep"] for k in flip_keys],
                [k["step_keystep"] for k in flip_keys],
                args.still_flip_guard,
                args.still_max_scale,
                args.still_ema_win,
                args.still_hysteresis,
                args.still_ratio
            )
        
        # 3. Sparse post-processing
        key_map = {}
        if 0 not in key_map: key_map[0] = "start_anchor"
        if (len(indices) - 1) not in key_map: key_map[len(indices) - 1] = "end_anchor"

        # Add flip keysteps
        for k in flip_keys:
            try:
                local_idx = steps.index(k["step_keystep"])
                key_map[local_idx] = k
            except ValueError: pass
            
        # Add still keysteps (lower priority; ignore overlaps with flips)
        for k in st_keys:
            try:
                local_idx = steps.index(k["step_keystep"])
                if local_idx not in key_map:
                    key_map[local_idx] = k
            except ValueError: pass

        # 3.1 Fill gaps larger than 15
        sorted_indices = sorted(key_map.keys())
        final_local_indices = []
        
        for i in range(len(sorted_indices) - 1):
            curr_idx = sorted_indices[i]
            next_idx = sorted_indices[i+1]
            final_local_indices.append(curr_idx)
            
            gap = next_idx - curr_idx
            if gap > args.sparse_gap_limit:
                mid_idx = curr_idx + gap // 2
                if mid_idx > curr_idx and mid_idx < next_idx:
                    final_local_indices.append(mid_idx)
        
        final_local_indices.append(sorted_indices[-1])
        unique_final = sorted(list(set(final_local_indices)))
        
        # 3.2 Special two-frame case: (0 + X) -> insert a fill frame
        if len(unique_final) == 2:
            start_i = unique_final[0]
            end_i = unique_final[1]
            # Only insert when there is room in between
            if end_i > start_i + 1:
                mid_idx = start_i + (end_i - start_i) // 2
                unique_final.insert(1, mid_idx)

        # 4. Build output records
        for local_idx in unique_final:
            if local_idx == 0: continue # Drop the Start frame
                
            if local_idx in key_map and isinstance(key_map[local_idx], dict):
                item = key_map[local_idx]
                item["episode_id"] = ep_id
                item["ep_name"] = frames[int(indices[local_idx])]["ep_name"]
            else:
                itype = "end" if local_idx == len(indices) - 1 else "gap_fill"
                item = make_full_item(frames, indices, steps, local_idx, item_type=itype)
                item["episode_id"] = ep_id
                item["ep_name"] = frames[int(indices[local_idx])]["ep_name"]
            
            # Fill missing fields
            global_idx = int(indices[local_idx])
            item["action"] = frames[global_idx]["action"]
            item["gripper"] = frames[global_idx]["gripper"]
            item["image"] = frames[global_idx]["image"]
            
            all_rows.append(item)

    # 5. Save results
    log(f"Extracted {len(all_rows)} steps total.")
    
    if all_rows:
        df = pd.DataFrame(all_rows)
        # Reorder columns
        cols_order = [
            "episode_id", "ep_name", "step_keystep", "type", 
            "idx_before", "idx_keystep", "step_before", 
            "gripper_before", "gripper_after", 
            "pos_delta_keystep", "rot_delta_keystep",
            "found_premove", "lookahead_used", "binarize_mode",
            "action", "gripper", "image"
        ]
        final_cols = [c for c in cols_order if c in df.columns]
        for c in df.columns:
            if c not in final_cols: final_cols.append(c)
        df = df[final_cols]
        
        # Save CSV
        df.to_csv(out_csv_path, index=False)
        log(f"Saved CSV to {out_csv_path}")

        # Generate summary.json
        type_counts = df["type"].value_counts().to_dict()
        summary = {
            "total_episodes": len(grouped),
            "total_extracted_frames": len(df),
            "breakdown": type_counts,
            "params": vars(args)
        }
        # Convert NumPy values in params into plain float/int types
        # Keep only basic fields for simplicity
        with open(out_json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log(f"Saved Summary to {out_json_path}")
        
    else:
        log("No steps extracted.", "WARN")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keysteps extractor for (Bridge / LIBERO-style) PKL datasets (gripper-settle + pre-move, CSV-only)

Core logic:
- flip: after gripper stabilization, select the frame right before the first frame whose instantaneous magnitude crosses the threshold
- still: select frames with unchanged gripper state, small short-window mean magnitude, and enough distance from flip keysteps
- gap_fill / backfill: fill sparse regions when keysteps are too sparse

This version keeps the validated keyframe selection logic from test_vision that reproduces the old CSV results, while removing image decoding and saving.

Added:
- cp clustering: cp events with interval < bad_cp_min_interval are merged into one cluster; only the last cp in the cluster is kept and marked as forced_cp
- fallback: if no keystep exists in the whole episode, force two backfill keysteps at the middle frame and the 11th frame from the end
- relaxed still rule: use a looser still threshold in regions sufficiently far from all flip keysteps
"""

import argparse, os, sys, json, pickle
from typing import Any, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------ utils ------------------------

def log(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}")

def find_change_points(bits: List[int]) -> List[int]:
    return [i for i in range(1, len(bits)) if bits[i] != bits[i-1]]

def l2(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.astype(np.float32)))

def ema_1d(x: np.ndarray, win: int) -> np.ndarray:
    """Simple EMA: effective window size is roughly win; return the input when win <= 1."""
    if win <= 1:
        return x.astype(np.float32)
    alpha = 2.0 / (win + 1.0)
    y = np.empty_like(x, dtype=np.float32)
    acc = float(x[0])
    for i in range(len(x)):
        acc = alpha * float(x[i]) + (1.0 - alpha) * acc
        y[i] = acc
    return y

def action_inst_mags(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous magnitude with action treated as delta/velocity."""
    if actions.ndim != 2 or actions.shape[1] < 6:
        T = actions.shape[0]
        return np.full(T, 1e9, dtype=np.float32), np.full(T, 1e9, dtype=np.float32)
    pos_mag = np.sqrt(np.sum(np.square(actions[:, :3].astype(np.float32)), axis=1))
    rot_mag = np.sqrt(np.sum(np.square(actions[:, 3:6].astype(np.float32)), axis=1))
    return pos_mag.astype(np.float32), rot_mag.astype(np.float32)

def window_action_magnitude(actions: np.ndarray) -> float:
    """Used to filter very short and low-magnitude false flips."""
    if actions.ndim != 2 or actions.shape[1] < 6 or actions.shape[0] == 0:
        return 999.0
    mean6 = np.mean(actions[:, :6], axis=0)
    return float(np.sum(np.abs(mean6)) * actions.shape[0])

def binarize_with_mode(x: float, mode: str) -> int:
    if mode == "gt05":
        return int(x > 0.5)
    elif mode == "gt0":
        return int(x > 0.0)
    else:
        return int(np.sign(x) > 0)

# ------------------------ Image utilities ------------------------

def to_uint8_img(arr: np.ndarray) -> np.ndarray:
    """Normalize any numeric array to [H, W, 3] uint8 for visualization."""
    a = np.asarray(arr).astype(np.float32)
    if a.size == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8) * 255.0
    a = a.clip(0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[-1] >= 3:
        return a[:, :, :3]
    if a.ndim == 2:
        return np.stack([a, a, a], axis=-1)
    a = np.squeeze(a)
    if a.ndim == 2:
        return np.stack([a, a, a], axis=-1)
    return np.zeros((64, 64, 3), dtype=np.uint8)

def _normalize_images_field(field: Any, T: int) -> List[Any]:
    """
    Normalize various image field formats into a length-T list for indexing.
    Supported inputs:
      - list/tuple with length possibly different from T
      - np.ndarray, treating shape[0] as the time dimension
      - dict, e.g. {"image": [...]} or multi-camera {"main": [...], "wrist": [...]}
      - a single string / array, repeated T times
      - None, replaced with empty-string placeholders
    """
    # dict: choose one sub-key ("image" / "main" / the first available key)
    if isinstance(field, dict):
        if "image" in field:
            field = field["image"]
        else:
            if len(field) > 0:
                first_key = sorted(field.keys())[0]
                field = field[first_key]
            else:
                return [""] * T

    # np.ndarray: treat the first dimension as time
    if isinstance(field, np.ndarray):
        if field.ndim >= 1:
            n = field.shape[0]
            if n >= T:
                return [field[i] for i in range(T)]
            elif n > 0:
                lst = [field[i] for i in range(n)]
                last = lst[-1]
                lst.extend([last] * (T - n))
                return lst
            else:
                return [""] * T
        else:
            return [field] * T

    # list / tuple
    if isinstance(field, (list, tuple)):
        if len(field) >= T:
            return list(field[:T])
        elif len(field) == 0:
            return [""] * T
        else:
            lst = list(field)
            last = lst[-1]
            lst.extend([last] * (T - len(lst)))
            return lst

    # None -> empty placeholders
    if field is None:
        return [""] * T

    # Any other type (str, a single path, etc.) is repeated T times
    return [field] * T


def flatten_pkl_libero(pkl_path: str, test: bool = False) -> List[Dict[str, Any]]:
    """
    Support Bridge / LIBERO-style data:
    episodes: List[Dict], where each episode contains at least:
      - "action": [T, >=7]
      - "image": List[T] or ndarray[T,...] / path, for the main camera
      - optional "image_gripper" or "gripper_image": List[T] / ndarray[T,...], for the gripper camera
      - optional "episode_name" / "task_name"
    """
    with open(pkl_path, "rb") as f:
        episodes = pickle.load(f)

    # Some datasets wrap episodes as {"episodes": [...]}; support that form too
    if isinstance(episodes, dict):
        if "episodes" in episodes:
            episodes = episodes["episodes"]
        else:
            episodes = [episodes]

    assert isinstance(episodes, list), "PKL must be a List[Dict] or a dict containing 'episodes'"
    if test:
        episodes = episodes[:10]

    rows: List[Dict[str, Any]] = []
    for epi, ep in enumerate(episodes):
        actions = np.asarray(ep["action"])
        assert actions.ndim == 2 and actions.shape[1] >= 7, "action must have shape [T,>=7]"
        T = actions.shape[0]

        # Main camera images
        raw_imgs_main = ep.get("image", None)
        imgs_main = _normalize_images_field(raw_imgs_main, T)

        # Gripper camera images (fallback to the main camera if missing)
        raw_imgs_grip = ep.get("image_gripper", ep.get("gripper_image", imgs_main))
        imgs_grip = _normalize_images_field(raw_imgs_grip, T)

        ep_name = ep.get("episode_name", ep.get("task_name", str(epi)))

        for t in range(T):
            rows.append({
                "episode_id": str(epi),
                "episode_name": str(ep_name),
                "step": t,
                "action": actions[t],
                "gripper": float(actions[t, -1]),
                "image": imgs_main[t],
                "image_gripper": imgs_grip[t],
            })
    return rows


# ------------------------ Threshold computation ------------------------

def compute_episode_thresholds(actions: np.ndarray,
                               flip_pos_pctl: float, flip_rot_pctl: float,
                               still_pos_pctl: float, still_rot_pctl: float,
                               flip_scale: float, still_scale: float,
                               flip_pos_min: float, flip_rot_min: float,
                               still_pos_max: float, still_rot_max: float) -> Dict[str, float]:
    """
    Estimate from per-episode instantaneous motion statistics:
      - flip_enter_*: thresholds for entering large motion (high percentiles)
      - still_*: thresholds for stillness (low percentiles)
    with lower/upper bounds and scaling factors applied.
    """
    pos_mag, rot_mag = action_inst_mags(actions)
    if not np.isfinite(pos_mag).all():
        pos_mag = np.nan_to_num(pos_mag, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(rot_mag).all():
        rot_mag = np.nan_to_num(rot_mag, nan=0.0, posinf=0.0, neginf=0.0)

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

# ------------------------ Gripper stabilization window ------------------------

def find_grip_settle_end(g_values: np.ndarray, grip_bits: List[int], cp: int,
                         new_state: int, settle_eps: float, settle_win: int, T: int) -> int:
    win = max(1, int(settle_win))
    if cp >= T:
        return T - 1
    dg = np.zeros(T, dtype=np.float32)
    dg[1:] = np.abs(g_values[1:] - g_values[:-1])
    for t in range(cp, T - win + 1):
        ok = True
        for u in range(t, t + win):
            if grip_bits[u] != new_state or dg[u] > settle_eps:
                ok = False
                break
        if ok:
            return t + win - 1
    return cp

# ------------------------ Flip: use instantaneous motion magnitude to detect onset ------------------------

def pick_keystep_after_grip_settle_libero(frames: List[Dict[str, Any]],
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
                                   pre_move_horizon: int) -> Tuple[int, int, float, float, bool]:
    """
    Select the flip keystep as the last low-motion frame before motion onset.

    - After the gripper change point cp, find when the gripper has actually settled.
    - Start scanning from max(cp + min_offset_after_flip, settle_end).
    - Use enter_pos / enter_rot as the thresholds for entering large motion:
        * if |v| stays below the threshold, treat it as still / slight motion and update last_small_t
        * stop once a frame first exceeds the threshold, which marks large-motion onset
    - keystep = last_small_t, i.e. the final still frame before the large motion
    """
    new_state = grip_bits[cp]
    T = len(indices)

    acts = np.stack([np.asarray(frames[int(i)]["action"], dtype=np.float32) for i in indices], axis=0)
    pos_mag, rot_mag = action_inst_mags(acts)

    settle_end = find_grip_settle_end(
        g_values, grip_bits, cp, new_state,
        settle_eps=grip_settle_eps,
        settle_win=grip_settle_win,
        T=T,
    )

    start_t = max(cp + max(1, int(min_offset_after_flip)), settle_end)
    if start_t >= T:
        start_t = T - 1
    end_t = min(cp + int(max_lookahead), T - 1)

    pos_thr = float(enter_pos)
    rot_thr = float(enter_rot)

    last_small_t = start_t
    found_premove = False

    t = start_t
    while t <= end_t:
        if grip_bits[t] != new_state:
            break

        pm = float(pos_mag[t])
        rm = float(rot_mag[t])

        if (pm < pos_thr) and (rm < rot_thr):
            last_small_t = t
            t += 1
            continue

        found_premove = True
        break

    ks_t = int(last_small_t)
    ks_idx = int(indices[ks_t])
    pos_at_ks = float(pos_mag[ks_t])
    rot_at_ks = float(rot_mag[ks_t])
    lookahead_used = ks_t - cp

    return ks_idx, lookahead_used, pos_at_ks, rot_at_ks, found_premove


# ------------------------ stillness ------------------------

def collect_still_keysteps_libero(frames: List[Dict[str, Any]],
                           indices: List[int],
                           steps: List[int],
                           actions: np.ndarray,
                           grip_bits: List[int],
                           mode_used: str,
                           pos_mag_thresh: float,
                           rot_mag_thresh: float,
                           win: int,
                           min_gap: int,
                           existing_steps: List[int],
                           flip_steps: List[int],
                           flip_guard: int,
                           max_scale: float,
                           ema_win: int,
                           hysteresis: float,
                           ratio_req: float,
                           start_step: int = 0,
                           cp_steps: Optional[List[int]] = None,
                           cp_guard: Optional[int] = None,
                           relax_dist: Optional[int] = None,
                           relax_scale: float = 1.0,
                           relax_ratio: Optional[float] = None) -> List[Dict[str, Any]]:

    T = actions.shape[0]
    if T == 0 or actions.ndim != 2 or actions.shape[1] < 6:
        return []

    pos_mag = np.sqrt(np.sum(np.square(actions[:, :3].astype(np.float32)), axis=1))
    rot_mag = np.sqrt(np.sum(np.square(actions[:, 3:6].astype(np.float32)), axis=1))

    if win <= 1:
        pos_mean, rot_mean = pos_mag, rot_mag
    else:
        k = np.ones(win, dtype=np.float32) / win
        pos_mean = np.convolve(pos_mag, k, mode="full")[:T]
        rot_mean = np.convolve(rot_mag, k, mode="full")[:T]

    def window_max(arr: np.ndarray, end_t: int, w: int) -> float:
        if end_t < w - 1:
            return float("inf")
        t0 = end_t - w + 1
        return float(np.max(arr[t0:end_t + 1]))

    used = set(int(s) for s in existing_steps)   # flip + existing still/backfill steps
    flip_set = set(int(s) for s in flip_steps)

    if cp_steps is None:
        cp_steps = []
    if cp_guard is None:
        cp_guard = 0
    cp_set = set(int(s) for s in cp_steps)

    chosen: List[Dict[str, Any]] = []
    last_still_step: Optional[int] = None
    w = max(1, int(win))

    for t in range(T):
        if t < w - 1:
            continue

        step_t = int(steps[t])

        # 0) Ignore still candidates near the beginning of the episode (e.g. first 10 frames)
        if step_t < start_step:
            continue

        # 1) Keep at least min_gap away from existing keysteps
        if any(abs(step_t - s) < min_gap for s in used):
            continue
        if (last_still_step is not None) and abs(step_t - last_still_step) < min_gap:
            continue

        # 2) Measure distance to the nearest flip keystep to determine whether this is far from key events
        if flip_set and (relax_dist is not None) and (relax_dist > 0):
            d_to_flip = min(abs(step_t - s) for s in flip_set)
        else:
            d_to_flip = float("inf")
        use_relaxed = (relax_dist is not None) and (d_to_flip >= relax_dist)

        # 3) still thresholds and ratio requirements: strict nearby, relaxed farther away
        pos_thr = pos_mag_thresh * (relax_scale if use_relaxed else 1.0)
        rot_thr = rot_mag_thresh * (relax_scale if use_relaxed else 1.0)
        # Prevent thresholds from growing without bound; cap them at max_scale
        pos_thr = min(pos_thr, pos_mag_thresh * max_scale)
        rot_thr = min(rot_thr, rot_mag_thresh * max_scale)

        ratio_thr = relax_ratio if (use_relaxed and (relax_ratio is not None)) else ratio_req
        pos_exit = float(pos_thr) * float(hysteresis)
        rot_exit = float(rot_thr) * float(hysteresis)

        # 4) Stay at least cp_guard away from all gripper change points
        if cp_guard > 0 and cp_set and any(abs(step_t - cs) <= cp_guard for cs in cp_set):
            continue

        # 5) Gripper state must remain constant within the window
        t0 = t - w + 1
        bit_ref = grip_bits[t0]
        if any(grip_bits[u] != bit_ref for u in range(t0, t + 1)):
            continue

        # 6) Most frames in the window must satisfy the pos/rot thresholds
        pos_ok = (pos_mag[t0:t + 1] <= pos_thr)
        rot_ok = (rot_mag[t0:t + 1] <= rot_thr)
        ratio_ok = float(np.sum(pos_ok & rot_ok)) / float(w)
        if ratio_ok < float(ratio_thr):
            continue

        # 7) The maximum motion in the window must stay small enough to reject obvious large motions
        if window_max(pos_mag, t, w) > (pos_thr * max_scale):
            continue
        if window_max(rot_mag, t, w) > (rot_thr * max_scale):
            continue

        # 8) The average magnitude should also stay below the exit threshold (hysteresis)
        if (pos_mean[t] > pos_exit) or (rot_mean[t] > rot_exit):
            continue

        # Record a still keystep once all conditions pass
        rel_b = max(0, t - 1)
        idx_b = int(indices[rel_b])
        idx_g = int(indices[t])
        gb = int(grip_bits[rel_b])
        ga = int(grip_bits[t])

        chosen.append({
            "type": "still",
            "cp": -1,
            "idx_before": idx_b,
            "idx_keystep": idx_g,
            "step_before": int(steps[rel_b]),
            "step_keystep": step_t,
            "gripper_before": gb,
            "gripper_after": ga,
            "pos_delta_keystep": float(pos_mean[t]),
            "rot_delta_keystep": float(rot_mean[t]),
            "lookahead_used": 0,
            "binarize_mode": mode_used,
            "found_premove": False,
        })
        used.add(step_t)
        last_still_step = step_t

    return chosen


# ------------------------ gap fill & backfill ------------------------

def select_most_stable_in_window(steps: List[int], actions: np.ndarray, g_values: np.ndarray,
                                 cur_step: int, gap: int, rot_weight: float, mode_used: str) -> int:
    last_step = steps[-1]
    right = min(cur_step + gap, last_step)
    if right <= cur_step:
        return cur_step

    def binarize(x: float) -> int:
        if mode_used == "gt05":
            return int(x > 0.5)
        elif mode_used == "gt0":
            return int(x > 0.0)
        else:
            return int(np.sign(x) >= 0)

    cur_rel = steps.index(cur_step)
    cur_state = binarize(g_values[cur_rel])
    best_t, best_score = None, None
    for t_step in range(cur_step + 1, right + 1):
        rel = steps.index(t_step)
        if rel - 1 < 0:
            continue
        if binarize(g_values[rel]) != cur_state:
            continue
        a_prev = actions[rel - 1]
        a_curr = actions[rel]
        dp = l2(a_curr[:3] - a_prev[:3])
        dr = l2(a_curr[3:6] - a_prev[3:6])
        score = dp + rot_weight * dr
        if (best_score is None) or (score < best_score):
            best_score, best_t = score, t_step
    return right if best_t is None else best_t

def nearest_valid_step(prefer_step: int, steps: List[int],
                       forbidden_steps: List[int], min_gap: int) -> Optional[int]:
    cands = sorted(steps, key=lambda s: (abs(s - prefer_step), s))
    for s in cands:
        if all(abs(s - f) >= min_gap for f in forbidden_steps):
            return s
    return None

def make_backfill_item(frames: List[Dict[str, Any]],
                       indices: List[int],
                       steps: List[int],
                       actions: np.ndarray,
                       g_values: np.ndarray,
                       mode_used: str,
                       step_target: int) -> Dict[str, Any]:
    rel = steps.index(int(step_target))
    idx_g = int(indices[rel])
    rel_b = max(0, rel - 1)
    idx_b = int(indices[rel_b])

    a_prev = np.asarray(frames[int(indices[rel_b])]["action"], dtype=np.float32)
    a_curr = np.asarray(frames[int(indices[rel])]["action"], dtype=np.float32)
    dp = l2(a_curr[:3] - a_prev[:3])
    dr = l2(a_curr[3:6] - a_prev[3:6])

    gb = binarize_with_mode(float(g_values[rel_b]), mode_used)
    ga = binarize_with_mode(float(g_values[rel]), mode_used)

    return {
        "type": "backfill",
        "cp": -1,
        "idx_before": int(indices[rel_b]),
        "idx_keystep": idx_g,
        "step_before": int(steps[rel_b]),
        "step_keystep": int(steps[rel]),
        "gripper_before": gb,
        "gripper_after": ga,
        "pos_delta_keystep": float(dp),
        "rot_delta_keystep": float(dr),
        "lookahead_used": 0,
        "binarize_mode": mode_used,
        "found_premove": False,
    }

# ------------------------ main ------------------------


# ------------------------ bridge-specific functions ------------------------

def flatten_pkl_bridge(pkl_path: str, test: bool = False) -> List[Dict[str, Any]]:
    with open(pkl_path, "rb") as f:
        episodes = pickle.load(f)
    assert isinstance(episodes, list), "PKL must be a List[Dict]"
    if test: episodes = episodes[:10]

    rows = []
    for epi, ep in enumerate(episodes):
        actions = np.asarray(ep["action"])
        assert actions.ndim == 2 and actions.shape[1] >= 7, "action must have shape [T,>=7]"
        T = actions.shape[0]
        imgs = ep.get("image", [""] * T)
        assert isinstance(imgs, (list, tuple)) and len(imgs) >= T
        for t in range(T):
            rows.append({
                "episode_id": str(epi),
                "step": t,
                "action": actions[t],
                "gripper": float(actions[t, -1]),
                "image": imgs[t],
            })
    return rows

def pick_keystep_after_grip_settle_bridge(frames: List[Dict[str, Any]],
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
                                   pre_move_horizon: int) -> Tuple[int, int, float, float, bool]:
    """
    Logic changes:
      - No longer take finite differences over actions; use EMA / mean of instantaneous magnitude instead.
      - Treat the first frame k that exceeds enter_* as motion onset, and use keystep = max(settle_end, k-1).
      - Track last_still_t using the looser exit_* thresholds (enter_* * hysteresis) to avoid jitter.
    """
    new_state = grip_bits[cp]
    T = len(indices)

    # Compute the instantaneous magnitude sequence for this episode once
    acts = np.stack([np.asarray(frames[int(i)]["action"], dtype=np.float32) for i in indices], axis=0)
    pos_mag, rot_mag = action_inst_mags(acts)
    pos_s = ema_1d(pos_mag, ema_win)
    rot_s = ema_1d(rot_mag, ema_win)

    settle_end = find_grip_settle_end(g_values, grip_bits, cp, new_state,
                                      settle_eps=grip_settle_eps, settle_win=grip_settle_win, T=T)
    search_start = min(max(cp + max(1, int(min_offset_after_flip)), settle_end), T-1)
    end_t = min(cp + int(max_lookahead), T - 1)

    enter_pos_thr, enter_rot_thr = float(enter_pos), float(enter_rot)
    exit_pos_thr  = enter_pos_thr * float(hysteresis)
    exit_rot_thr  = enter_rot_thr * float(hysteresis)

    last_still_t = search_start
    found_premove = False

    t = search_start
    while t <= end_t:
        if grip_bits[t] != new_state:
            break
        horizon_end = min(t + int(pre_move_horizon), T - 1)

        # Find the first frame in [t, horizon_end] that exceeds the enter threshold
        first_big_k = None
        for k in range(t, horizon_end + 1):
            if (pos_s[k] > enter_pos_thr) or (rot_s[k] > enter_rot_thr):
                first_big_k = k
                break

        if first_big_k is not None:
            ks_t = max(settle_end, first_big_k - 1)
            ks_idx = int(indices[ks_t])
            # Record the instantaneous magnitude at the keystep frame
            pos_at_ks = float(pos_mag[ks_t])
            rot_at_ks = float(rot_mag[ks_t])
            found_premove = True
            return ks_idx, (ks_t - cp), pos_at_ks, rot_at_ks, found_premove

        # Before entering the threshold: update last_still_t using the looser exit threshold
        if (pos_s[t] <= exit_pos_thr) and (rot_s[t] <= exit_rot_thr):
            last_still_t = t
        t += 1

    # If no onset is found, fall back to the last frame still within the exit threshold
    ks_t = int(last_still_t)
    ks_idx = int(indices[ks_t])
    pos_at_ks = float(pos_mag[ks_t])
    rot_at_ks = float(rot_mag[ks_t])
    return ks_idx, (ks_t - cp), pos_at_ks, rot_at_ks, found_premove

def collect_still_keysteps_bridge(frames: List[Dict[str, Any]],
                           indices: List[int],
                           steps: List[int],
                           actions: np.ndarray,
                           grip_bits: List[int],
                           mode_used: str,
                           pos_mag_thresh: float,
                           rot_mag_thresh: float,
                           win: int,
                           min_gap: int,
                           existing_steps: List[int],
                           flip_steps: List[int],
                           flip_guard: int,
                           max_scale: float,
                           ema_win: int,
                           hysteresis: float,
                           ratio_req: float) -> List[Dict[str, Any]]:

    T = actions.shape[0]
    if T == 0 or actions.ndim != 2 or actions.shape[1] < 6:
        return []

    # Per-frame instantaneous motion magnitude
    pos_mag = np.sqrt(np.sum(np.square(actions[:, :3].astype(np.float32)), axis=1))
    rot_mag = np.sqrt(np.sum(np.square(actions[:, 3:6].astype(np.float32)), axis=1))

    # Window mean for stability estimation; EMA would behave similarly here
    if win <= 1:
        pos_mean, rot_mean = pos_mag, rot_mag
    else:
        k = np.ones(win, dtype=np.float32) / win
        pos_mean = np.convolve(pos_mag, k, mode="full")[:T]
        rot_mean = np.convolve(rot_mag, k, mode="full")[:T]

    # Hysteresis threshold (a looser exit threshold)
    pos_exit = float(pos_mag_thresh) * float(hysteresis)
    rot_exit = float(rot_mag_thresh) * float(hysteresis)

    def window_max(arr: np.ndarray, end_t: int, w: int) -> float:
        if end_t < w - 1: return float("inf")
        t0 = end_t - w + 1
        return float(np.max(arr[t0:end_t+1]))

    used = set(int(s) for s in existing_steps)
    flip_set = set(int(s) for s in flip_steps)
    chosen: List[Dict[str, Any]] = []
    last_still_step: Optional[int] = None
    w = max(1, int(win))

    for t in range(T):
        if t < w - 1:
            continue

        step_t = int(steps[t])

        # Keep distance from already selected keysteps
        if any(abs(step_t - s) < min_gap for s in used):
            continue
        if (last_still_step is not None) and abs(step_t - last_still_step) < min_gap:
            continue
        # Avoid the protection band around flips
        if any(abs(step_t - fs) <= flip_guard for fs in flip_set):
            continue

        # Require the gripper state to stay constant within the window
        t0 = t - w + 1
        bit_ref = grip_bits[t0]
        if any(grip_bits[u] != bit_ref for u in range(t0, t + 1)):
            continue

        # === Core three-part check ===
        # (1) Ratio criterion: at least ratio_req of frames must satisfy both pos/rot thresholds
        pos_ok = (pos_mag[t0:t+1] <= pos_mag_thresh)
        rot_ok = (rot_mag[t0:t+1] <= rot_mag_thresh)
        ratio_ok = float(np.sum(pos_ok & rot_ok)) / float(w)
        if ratio_ok < float(ratio_req):
            continue

        # (2) Maximum-value guardrail: allow jitter, but reject sharp spikes
        if window_max(pos_mag, t, w) > (pos_mag_thresh * max_scale):
            continue
        if window_max(rot_mag, t, w) > (rot_mag_thresh * max_scale):
            continue

        # (3) The mean only needs to stay below the exit threshold; ratio + guardrails handle stricter filtering
        if (pos_mean[t] > pos_exit) or (rot_mean[t] > rot_exit):
            continue

        # Record one still keystep
        rel_b = max(0, t - 1)
        idx_b = int(indices[rel_b])
        idx_g = int(indices[t])
        gb = int(grip_bits[rel_b]); ga = int(grip_bits[t])

        chosen.append({
            "type": "still",
            "cp": -1,
            "idx_before": idx_b,
            "idx_keystep": idx_g,
            "step_before": int(steps[rel_b]),
            "step_keystep": step_t,
            "gripper_before": gb,
            "gripper_after": ga,
            # Use the window-end mean to represent stillness
            "pos_delta_keystep": float(pos_mean[t]),
            "rot_delta_keystep": float(rot_mean[t]),
            "lookahead_used": 0,
            "binarize_mode": mode_used,
            "found_premove": False,
        })
        used.add(step_t)
        last_still_step = step_t

    return chosen

# ------------------------ unified runner ------------------------



BASE_DEFAULTS = {
    'dataset': '',
    'out_dir': '/remote-home/jinminghao/structvla/documents/keysteps_custom',
    'min_changes': 0,
    'max_lookahead': 12,
    'min_offset_after_flip': 1,
    'pre_move_horizon': 2,
    'grip_settle_eps': 0.05,
    'grip_settle_win': 2,
    'min_window': 2,
    'mag_thresh': 0.2,
    'flip_pos_pctl': 70.0,
    'flip_rot_pctl': 70.0,
    'flip_scale': 0.9,
    'flip_pos_min': 0.05,
    'flip_rot_min': 0.05,
    'flip_ema_win': 2,
    'flip_hysteresis': 0.7,
    'still_pos_pctl': 35.0,
    'still_rot_pctl': 35.0,
    'still_scale': 1.2,
    'still_pos_max': 0.5,
    'still_rot_max': 0.5,
    'still_win': 4,
    'still_min_gap': 10,
    'still_flip_guard': 1,
    'still_max_scale': 3.0,
    'still_ema_win': 3,
    'still_hysteresis': 1.2,
    'still_ratio': 0.5,
    'still_cp_guard': 0,
    'still_start_offset': 0,
    'still_relax_dist': 0,
    'still_relax_scale': 1.0,
    'still_relax_ratio': 0.0,
    'bad_cp_min_interval': 0,
    'skip_bad_cp_episode': False,
    'max_cp_per_episode': 0,
    'max_keystep_gap': 10,
    'rot_weight': 1.0,
    'no_gap_fill': True,
    'backfill_min_gap': 10,
    'backfill_tail_offset': 3,
    'no_still_keys': False,
    'dump_actions_on_empty': False,
}

DATASET_PRESETS = {
    'simpler': {
        'logic_profile': 'bridge',
        'dataset': '',
        'out_dir': '',
        'overrides': {
            'min_changes': 0,
            'max_lookahead': 12,
            'min_offset_after_flip': 1,
            'pre_move_horizon': 2,
            'grip_settle_eps': 0.05,
            'grip_settle_win': 2,
            'min_window': 2,
            'mag_thresh': 0.2,
            'flip_pos_pctl': 70.0,
            'flip_rot_pctl': 70.0,
            'flip_scale': 0.9,
            'flip_pos_min': 0.05,
            'flip_rot_min': 0.05,
            'flip_ema_win': 2,
            'flip_hysteresis': 0.7,
            'still_pos_pctl': 35.0,
            'still_rot_pctl': 35.0,
            'still_scale': 1.2,
            'still_pos_max': 0.5,
            'still_rot_max': 0.5,
            'still_win': 4,
            'still_min_gap': 10,
            'still_flip_guard': 1,
            'still_max_scale': 3.0,
            'still_ema_win': 3,
            'still_hysteresis': 1.2,
            'still_ratio': 0.5,
            'max_keystep_gap': 10,
            'rot_weight': 1.0,
            'no_gap_fill': True,
            'backfill_min_gap': 10,
            'backfill_tail_offset': 3,
            'no_still_keys': False,
            'dump_actions_on_empty': False,
            'still_cp_guard': 0,
            'still_start_offset': 0,
            'still_relax_dist': 0,
            'still_relax_scale': 1.0,
            'still_relax_ratio': 0.0,
            'bad_cp_min_interval': 0,
            'skip_bad_cp_episode': False,
            'max_cp_per_episode': 0,
        },
    },
    'libero': {
        'logic_profile': 'libero',
        'dataset': '',
        'out_dir': '',
        'overrides': {
            'min_changes': 0,
            'max_lookahead': 12,
            'min_offset_after_flip': 1,
            'pre_move_horizon': 2,
            'grip_settle_eps': 0.05,
            'grip_settle_win': 2,
            'min_window': 2,
            'mag_thresh': 0.2,
            'flip_pos_pctl': 80.0,
            'flip_rot_pctl': 80.0,
            'flip_scale': 0.9,
            'flip_pos_min': 0.04,
            'flip_rot_min': 0.04,
            'flip_ema_win': 2,
            'flip_hysteresis': 0.7,
            'still_pos_pctl': 25.0,
            'still_rot_pctl': 25.0,
            'still_scale': 1.0,
            'still_pos_max': 999.0,
            'still_rot_max': 999.0,
            'still_win': 4,
            'still_min_gap': 20,
            'still_flip_guard': 20,
            'still_max_scale': 1.5,
            'still_ema_win': 3,
            'still_hysteresis': 1.5,
            'still_ratio': 0.5,
            'still_cp_guard': 20,
            'still_start_offset': 20,
            'still_relax_dist': 40,
            'still_relax_scale': 1.5,
            'still_relax_ratio': 0.4,
            'bad_cp_min_interval': 20,
            'skip_bad_cp_episode': False,
            'max_cp_per_episode': 20,
            'max_keystep_gap': 999,
            'rot_weight': 1.0,
            'no_gap_fill': True,
            'backfill_min_gap': 10,
            'backfill_tail_offset': 10,
            'no_still_keys': False,
            'dump_actions_on_empty': False,
        },
    },
    'other': {
        'logic_profile': 'bridge',
        'dataset': '',
        'out_dir': '/remote-home/jinminghao/structvla/documents/keysteps_custom',
        'overrides': {},
    },
}


def collect_explicit_args(argv: List[str]) -> set[str]:
    explicit = set()
    for token in argv:
        if token == '--':
            break
        if not token.startswith('--'):
            continue
        name = token[2:].split('=', 1)[0].replace('-', '_')
        explicit.add(name)
    return explicit


def apply_dataset_preset(args: argparse.Namespace, explicit: set[str]) -> argparse.Namespace:
    preset = DATASET_PRESETS.get(args.datasets, DATASET_PRESETS['other'])
    if args.logic_profile is None:
        args.logic_profile = preset['logic_profile']
    if 'dataset' not in explicit and preset.get('dataset'):
        args.dataset = preset['dataset']
    if 'out_dir' not in explicit and preset.get('out_dir'):
        args.out_dir = preset['out_dir']
    for key, value in preset['overrides'].items():
        if key not in explicit:
            setattr(args, key, value)
    return args


def build_parser() -> argparse.ArgumentParser:
    d = BASE_DEFAULTS
    ap = argparse.ArgumentParser(description='Single-file keystep extractor with dataset-driven presets.')
    ap.add_argument('--datasets', type=str, default='other',
                    help='Dataset name. simpler/libero inject validated defaults; any other name uses generic defaults.')
    ap.add_argument('--logic_profile', choices=['bridge', 'libero'], default=None,
                    help='Optional logic override for advanced usage. If unset, inferred from --datasets.')
    ap.add_argument('--dataset', type=str, default=d['dataset'])
    ap.add_argument('--out_dir', type=str, default=d['out_dir'])
    ap.add_argument('--min_changes', type=int, default=d['min_changes'])
    ap.add_argument('--test', action='store_true')
    for name in ['max_lookahead','min_offset_after_flip','pre_move_horizon','grip_settle_win','min_window','flip_ema_win','still_win','still_min_gap','still_flip_guard','still_ema_win','max_keystep_gap','backfill_min_gap','backfill_tail_offset','still_cp_guard','still_start_offset','still_relax_dist','bad_cp_min_interval','max_cp_per_episode']:
        ap.add_argument(f'--{name}', type=int, default=d[name])
    for name in ['grip_settle_eps','mag_thresh','flip_pos_pctl','flip_rot_pctl','flip_scale','flip_pos_min','flip_rot_min','flip_hysteresis','still_pos_pctl','still_rot_pctl','still_scale','still_pos_max','still_rot_max','still_max_scale','still_hysteresis','still_ratio','rot_weight','still_relax_scale','still_relax_ratio']:
        ap.add_argument(f'--{name}', type=float, default=d[name])
    ap.add_argument('--no_still_keys', action='store_true', default=d['no_still_keys'])
    ap.add_argument('--no_gap_fill', action='store_true', default=d['no_gap_fill'])
    ap.add_argument('--dump_actions_on_empty', action='store_true', default=d['dump_actions_on_empty'])
    ap.add_argument('--skip_bad_cp_episode', action='store_true', default=d['skip_bad_cp_episode'])
    return ap

def run_bridge_profile(args):
    frames = flatten_pkl_bridge(args.dataset, test=args.test)
    grouped: Dict[str, List[int]] = {}
    for i, fr in enumerate(frames):
        grouped.setdefault(fr['episode_id'], []).append(i)
    log(f'Episodes detected: {len(grouped)}')
    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    for ep_id, indices in tqdm(grouped.items(), desc='Episodes'):
        g_values = np.array([float(frames[i]['gripper']) for i in indices], dtype=np.float32)
        steps = [int(frames[i]['step']) for i in indices]
        actions = np.stack([np.asarray(frames[i]['action'], dtype=np.float32) for i in indices], axis=0)
        attempts = [('gt05',(g_values>0.5).astype(int)),('gt0',(g_values>0.0).astype(int)),('sign',(np.sign(g_values)>=0).astype(int))]
        mode_used, bits = None, None
        for name, arr in attempts:
            if len(arr) >= 2 and not np.all(arr[1:] == arr[:-1]):
                mode_used, bits = name, arr
                break
        if bits is None:
            mode_used, bits = 'sign', attempts[-1][1]
        grip_bits = bits.tolist()
        cps = find_change_points(grip_bits) if len(grip_bits) >= 2 else []
        if len(cps) < args.min_changes:
            cps = []
        th = compute_episode_thresholds(actions, args.flip_pos_pctl, args.flip_rot_pctl, args.still_pos_pctl, args.still_rot_pctl, args.flip_scale, args.still_scale, args.flip_pos_min, args.flip_rot_min, args.still_pos_max, args.still_rot_max)
        first_idx_global = int(indices[0])
        first_step = int(frames[first_idx_global]['step'])
        action_first = list(map(float, np.asarray(frames[first_idx_global]['action']).tolist()))
        flip_keysteps: List[Dict[str, Any]] = []
        for cp in cps:
            before_rel = cp - 1
            if before_rel < 0:
                continue
            ks_idx_global, lookahead_used, pos_at_ks, rot_at_ks, found_premove = pick_keystep_after_grip_settle_bridge(
                frames, indices, cp, grip_bits, g_values, th['flip_enter_pos'], th['flip_enter_rot'], args.flip_hysteresis,
                args.flip_ema_win, args.max_lookahead, args.min_offset_after_flip, args.grip_settle_eps, args.grip_settle_win,
                args.pre_move_horizon,
            )
            step_before = int(frames[int(indices[before_rel])]['step'])
            step_keystep = int(frames[ks_idx_global]['step'])
            window_len = max(1, step_keystep - step_before + 1)
            if window_len <= max(args.min_window, 1):
                ks_rel = steps.index(step_keystep)
                win_actions = actions[before_rel: ks_rel + 1, :]
                if window_action_magnitude(win_actions) < args.mag_thresh:
                    continue
            flip_keysteps.append({'type':'flip','cp':int(cp),'idx_before':int(indices[before_rel]),'idx_keystep':int(ks_idx_global),'step_before':int(step_before),'step_keystep':int(step_keystep),'gripper_before':int(grip_bits[cp-1]),'gripper_after':int(grip_bits[cp]),'pos_delta_keystep':float(pos_at_ks),'rot_delta_keystep':float(rot_at_ks),'lookahead_used':int(lookahead_used),'binarize_mode':mode_used,'found_premove':bool(found_premove)})
        if not args.no_still_keys:
            flip_keysteps.extend(collect_still_keysteps_bridge(frames=frames, indices=indices, steps=steps, actions=actions, grip_bits=grip_bits, mode_used=mode_used, pos_mag_thresh=th['still_pos'], rot_mag_thresh=th['still_rot'], win=max(1, int(args.still_win)), min_gap=max(1, int(args.still_min_gap)), existing_steps=[k['step_keystep'] for k in flip_keysteps], flip_steps=[k['step_keystep'] for k in flip_keysteps], flip_guard=int(args.still_flip_guard), max_scale=float(args.still_max_scale), ema_win=int(args.still_ema_win), hysteresis=float(args.still_hysteresis), ratio_req=float(args.still_ratio)))
        all_keysteps = sorted(flip_keysteps, key=lambda x: x['step_keystep'])
        selected_steps = [int(k['step_keystep']) for k in all_keysteps]
        min_gap = int(args.backfill_min_gap)
        first_step_in_ep = int(steps[0]); last_step_in_ep = int(steps[-1])
        def append_if_valid(prefer_step: int, forbid: List[int], collected: List[Dict[str, Any]]):
            cand = nearest_valid_step(prefer_step, steps, forbid, min_gap)
            if cand is not None:
                item = make_backfill_item(frames, indices, steps, actions, g_values, mode_used, cand)
                collected.append(item); forbid.append(int(cand))
        backfills: List[Dict[str, Any]] = []
        if len(selected_steps) == 0:
            mid_pref = (first_step_in_ep + last_step_in_ep) // 2
            tail_pref = max(first_step_in_ep, last_step_in_ep - int(args.backfill_tail_offset))
            forbid: List[int] = []
            append_if_valid(mid_pref, forbid, backfills)
            append_if_valid(tail_pref, [k['step_keystep'] for k in backfills], backfills)
        elif len(selected_steps) == 1:
            k0 = selected_steps[0]; mid = (first_step_in_ep + last_step_in_ep) // 2; forbid = [k0]
            if k0 <= mid:
                append_if_valid(max(first_step_in_ep, last_step_in_ep - int(args.backfill_tail_offset)), forbid[:], backfills)
            else:
                append_if_valid((first_step_in_ep + k0) // 2, forbid[:], backfills)
        if backfills:
            all_keysteps = sorted(all_keysteps + backfills, key=lambda x: x['step_keystep'])
        for item in all_keysteps:
            idx_before = item['idx_before']; idx_ks = item['idx_keystep']
            action_before = list(map(float, np.asarray(frames[idx_before]['action']).tolist()))
            action_ks = list(map(float, np.asarray(frames[idx_ks]['action']).tolist()))
            manifest_rows.append({'episode_id':ep_id,'keystep_type':item['type'],'change_point':int(item['cp']),'idx_first':int(first_idx_global),'idx_before':int(idx_before),'idx_keystep':int(idx_ks),'step_first':int(first_step),'step_before':int(item['step_before']),'step_keystep':int(item['step_keystep']),'gripper_before':int(item['gripper_before']),'gripper_after':int(item['gripper_after']),'action_first':json.dumps(action_first),'action_before':json.dumps(action_before),'action_keystep':json.dumps(action_ks),'pos_delta_keystep':float(item['pos_delta_keystep']),'rot_delta_keystep':float(item['rot_delta_keystep']),'lookahead_used':int(item['lookahead_used']),'binarize_mode':item['binarize_mode'],'found_premove':bool(item.get('found_premove',False)),'flip_enter_pos':th['flip_enter_pos'],'flip_enter_rot':th['flip_enter_rot'],'still_pos_thr':th['still_pos'],'still_rot_thr':th['still_rot']})
    manifest_path = os.path.join(out_root, 'triplets_manifest.csv')
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    with open(os.path.join(out_root, 'summary.json'), 'w') as f:
        json.dump({
            'datasets': args.datasets,
            'logic_profile': args.logic_profile,
            'dataset': args.dataset,
            'episodes_detected': len(grouped),
            'triplets': len(manifest_rows),
            'out_root': out_root,
            'test_mode': args.test,
        }, f, ensure_ascii=False, indent=2)
    log(f'CSV in: {manifest_path}')


def run_libero_profile(args):
    frames = flatten_pkl_libero(args.dataset, test=args.test)
    grouped: Dict[str, List[int]] = {}
    for i, fr in enumerate(frames):
        grouped.setdefault(fr['episode_id'], []).append(i)
    log(f'Episodes detected: {len(grouped)}')
    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    for ep_id, indices in tqdm(grouped.items(), desc='Episodes'):
        g_values = np.array([float(frames[i]['gripper']) for i in indices], dtype=np.float32)
        steps = [int(frames[i]['step']) for i in indices]
        actions = np.stack([np.asarray(frames[i]['action'], dtype=np.float32) for i in indices], axis=0)
        ep_name = frames[indices[0]].get('episode_name', str(ep_id))
        attempts = [('gt05',(g_values>0.5).astype(int)),('gt0',(g_values>0.0).astype(int)),('sign',(np.sign(g_values)>=0).astype(int))]
        mode_used, bits = None, None
        for name, arr in attempts:
            if len(arr) >= 2 and not np.all(arr[1:] == arr[:-1]):
                mode_used, bits = name, arr
                break
        if bits is None:
            mode_used, bits = 'sign', attempts[-1][1]
        grip_bits = bits.tolist()
        cps = find_change_points(grip_bits) if len(grip_bits) >= 2 else []
        if len(cps) < args.min_changes:
            cps = []
        raw_cps = cps.copy(); forced_cps = set()
        if cps:
            clusters: List[List[int]] = []
            cur_cluster = [cps[0]]
            for cp_idx in cps[1:]:
                if steps[cp_idx] - steps[cur_cluster[-1]] <= args.bad_cp_min_interval:
                    cur_cluster.append(cp_idx)
                else:
                    clusters.append(cur_cluster); cur_cluster = [cp_idx]
            clusters.append(cur_cluster)
            cps_new: List[int] = []
            for cluster in clusters:
                if len(cluster) == 1:
                    cps_new.append(cluster[0])
                else:
                    last_cp = cluster[-1]; cps_new.append(last_cp); forced_cps.add(last_cp)
            cps = cps_new
        th = compute_episode_thresholds(actions, args.flip_pos_pctl, args.flip_rot_pctl, args.still_pos_pctl, args.still_rot_pctl, args.flip_scale, args.still_scale, args.flip_pos_min, args.flip_rot_min, args.still_pos_max, args.still_rot_max)
        if raw_cps and args.max_cp_per_episode > 0 and len(raw_cps) > args.max_cp_per_episode and args.skip_bad_cp_episode:
            continue
        first_idx_global = int(indices[0]); first_step = int(frames[first_idx_global]['step'])
        action_first = list(map(float, np.asarray(frames[first_idx_global]['action']).tolist()))
        flip_keysteps: List[Dict[str, Any]] = []
        for cp in cps:
            before_rel = cp - 1
            if before_rel < 0:
                continue
            ks_idx_global, lookahead_used, pos_at_ks, rot_at_ks, found_premove = pick_keystep_after_grip_settle_libero(frames, indices, cp, grip_bits, g_values, th['flip_enter_pos'], th['flip_enter_rot'], args.flip_hysteresis, args.flip_ema_win, args.max_lookahead, args.min_offset_after_flip, args.grip_settle_eps, args.grip_settle_win, args.pre_move_horizon)
            step_before = int(frames[int(indices[before_rel])]['step']); step_keystep = int(frames[ks_idx_global]['step'])
            window_len = max(1, step_keystep - step_before + 1)
            is_forced_cp = cp in forced_cps
            if (not is_forced_cp) and window_len <= max(args.min_window, 1):
                ks_rel = steps.index(step_keystep); win_actions = actions[before_rel: ks_rel + 1, :]
                if window_action_magnitude(win_actions) < args.mag_thresh:
                    continue
            flip_keysteps.append({'type':'flip','cp':int(cp),'idx_before':int(indices[before_rel]),'idx_keystep':int(ks_idx_global),'step_before':int(step_before),'step_keystep':int(step_keystep),'gripper_before':int(grip_bits[cp-1]),'gripper_after':int(grip_bits[cp]),'pos_delta_keystep':float(pos_at_ks),'rot_delta_keystep':float(rot_at_ks),'lookahead_used':int(lookahead_used),'binarize_mode':mode_used,'found_premove':bool(found_premove)})
        if not args.no_still_keys:
            flip_step_list = [k['step_keystep'] for k in flip_keysteps]
            cp_steps = [steps[cp] for cp in cps] if cps else []
            still_start_step = int(steps[0]) + int(args.still_start_offset)
            flip_keysteps.extend(collect_still_keysteps_libero(frames=frames, indices=indices, steps=steps, actions=actions, grip_bits=grip_bits, mode_used=mode_used, pos_mag_thresh=th['still_pos'], rot_mag_thresh=th['still_rot'], win=max(1, int(args.still_win)), min_gap=max(1, int(args.still_min_gap)), existing_steps=flip_step_list, flip_steps=flip_step_list, flip_guard=int(args.still_flip_guard), max_scale=float(args.still_max_scale), ema_win=int(args.still_ema_win), hysteresis=float(args.still_hysteresis), ratio_req=float(args.still_ratio), start_step=still_start_step, cp_steps=cp_steps, cp_guard=int(args.still_cp_guard), relax_dist=int(args.still_relax_dist) if args.still_relax_dist > 0 else None, relax_scale=float(args.still_relax_scale), relax_ratio=float(args.still_relax_ratio)))
        all_keysteps = sorted(flip_keysteps, key=lambda x: x['step_keystep'])
        selected_steps = [int(k['step_keystep']) for k in all_keysteps]
        min_gap = int(args.backfill_min_gap)
        first_step_in_ep = int(steps[0]); last_step_in_ep = int(steps[-1])
        def append_if_valid(prefer_step: int, forbid: List[int], collected: List[Dict[str, Any]]):
            cand = nearest_valid_step(prefer_step, steps, forbid, min_gap)
            if cand is not None:
                item = make_backfill_item(frames, indices, steps, actions, g_values, mode_used, cand)
                collected.append(item); forbid.append(int(cand))
        backfills: List[Dict[str, Any]] = []
        if len(selected_steps) == 0:
            mid_pref = (first_step_in_ep + last_step_in_ep) // 2
            tail_pref = last_step_in_ep - 10
            if tail_pref < first_step_in_ep: tail_pref = first_step_in_ep
            forbid: List[int] = []
            append_if_valid(mid_pref, forbid, backfills)
            append_if_valid(tail_pref, [k['step_keystep'] for k in backfills], backfills)
        elif len(selected_steps) == 1:
            k0 = selected_steps[0]; mid = (first_step_in_ep + last_step_in_ep) // 2; forbid = [k0]
            if k0 <= mid:
                append_if_valid(max(first_step_in_ep, last_step_in_ep - int(args.backfill_tail_offset)), forbid[:], backfills)
            else:
                append_if_valid((first_step_in_ep + k0) // 2, forbid[:], backfills)
        if backfills:
            all_keysteps = sorted(all_keysteps + backfills, key=lambda x: x['step_keystep'])
        for item in all_keysteps:
            idx_before = item['idx_before']; idx_ks = item['idx_keystep']
            action_before = list(map(float, np.asarray(frames[idx_before]['action']).tolist()))
            action_ks = list(map(float, np.asarray(frames[idx_ks]['action']).tolist()))
            manifest_rows.append({'episode_id':ep_id,'episode_name':ep_name,'keystep_type':item['type'],'change_point':int(item['cp']),'idx_first':int(first_idx_global),'idx_before':int(idx_before),'idx_keystep':int(idx_ks),'step_first':int(first_step),'step_before':int(item['step_before']),'step_keystep':int(item['step_keystep']),'gripper_before':int(item['gripper_before']),'gripper_after':int(item['gripper_after']),'action_first':json.dumps(action_first),'action_before':json.dumps(action_before),'action_keystep':json.dumps(action_ks),'pos_delta_keystep':float(item['pos_delta_keystep']),'rot_delta_keystep':float(item['rot_delta_keystep']),'lookahead_used':int(item['lookahead_used']),'binarize_mode':item['binarize_mode'],'found_premove':bool(item.get('found_premove',False)),'flip_enter_pos':th['flip_enter_pos'],'flip_enter_rot':th['flip_enter_rot'],'still_pos_thr':th['still_pos'],'still_rot_thr':th['still_rot']})
    manifest_path = os.path.join(out_root, 'triplets_manifest.csv')
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    with open(os.path.join(out_root, 'summary.json'), 'w') as f:
        json.dump({
            'datasets': args.datasets,
            'logic_profile': args.logic_profile,
            'dataset': args.dataset,
            'episodes_detected': len(grouped),
            'triplets': len(manifest_rows),
            'out_root': out_root,
            'test_mode': args.test,
        }, f, ensure_ascii=False, indent=2)
    log(f'CSV in: {manifest_path}')


def main():
    parser = build_parser()
    args = parser.parse_args()
    explicit = collect_explicit_args(sys.argv[1:])
    args = apply_dataset_preset(args, explicit)
    if not args.dataset:
        parser.error('`--dataset` is required when no dataset preset path is available.')
    if args.logic_profile == 'bridge':
        run_bridge_profile(args)
    else:
        run_libero_profile(args)


if __name__ == '__main__':
    main()

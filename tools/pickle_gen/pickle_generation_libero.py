import os
import os.path as osp
import pickle
from tqdm import tqdm
import numpy as np
import sys
import argparse
import glob  # NEW: for matching libero_all_codes_*

# Project-specific paths
PROJECT_ROOT = "/remote-home/jinminghao"
sys.path.append(f"{PROJECT_ROOT}/structvla")

# Import normalization utility
from train.dataset.normalize_pi0 import RunningStats, save

def sort_by_int(filename):
    return int(os.path.splitext(filename)[0])

# NEW: helper to collect all libero_all_codes_* / libero_all_gripper_codes_* dirs
def get_code_dirs(dataset_path, base_prefix):
    """
    Find all directories under dataset_path matching:
      {base_prefix}_*
    and sort them by trailing number in descending order, for example:
      libero_all_codes_300, libero_all_codes_200
    => [300, 200]
    """
    pattern = osp.join(dataset_path, f"{base_prefix}_*")
    dirs = [d for d in glob.glob(pattern) if osp.isdir(d)]

    def suffix_num(path):
        name = osp.basename(path)  # e.g. libero_all_codes_300
        try:
            suffix = name.split("_")[-1]
            return int(suffix)
        except ValueError:
            return 0

    # Higher numeric suffix comes first: 300 > 200
    return sorted(dirs, key=suffix_num, reverse=True)

def main(dataset_path, output_path, normalizer_path, output_filename):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(normalizer_path, exist_ok=True)

    language_dir = osp.join(dataset_path, "libero_all")
    #vq_dir = osp.join(dataset_path, "libero_all_codes_200")
    #gripper_vq_dir = osp.join(dataset_path, "libero_all_gripper_codes_200")

    # NEW: Support arbitrary libero_all_codes_* / libero_all_gripper_codes_*
    vq_dirs = get_code_dirs(dataset_path, "libero_all_codes")
    gripper_vq_dirs = get_code_dirs(dataset_path, "libero_all_gripper_codes")

    if not vq_dirs:
        raise ValueError(f"No libero_all_codes_* directories found under {dataset_path}")
    if not gripper_vq_dirs:
        raise ValueError(f"No libero_all_gripper_codes_* directories found under {dataset_path}")

    print("Using image code dirs (priority order):")
    for d in vq_dirs:
        print("  ", d)
    print("Using gripper code dirs (priority order):")
    for d in gripper_vq_dirs:
        print("  ", d)


    min_frames = 8
    result_file = []

    print("Loading scenes from:", language_dir)
    for scene in tqdm(os.listdir(language_dir)):
        instr_file = osp.join(language_dir, scene, "instruction.txt")
        if not osp.exists(instr_file):
            continue
        with open(instr_file, "r") as f:
            text = f.read()

        # Load action sequences
        action_folder = osp.join(language_dir, scene, "actions")
        if not osp.exists(action_folder):
            continue
        action_files = [osp.join(action_folder, file) for file in sorted(os.listdir(action_folder), key=sort_by_int)]
        if len(action_files) < min_frames:
            continue
        action = [np.load(a) for a in action_files]

        # Load image tokens
        # img_dir = osp.join(vq_dir, scene)
        # if not osp.exists(img_dir):
        #     continue
        # img_files = [osp.join(img_dir, file) for file in sorted(os.listdir(img_dir), key=sort_by_int)]

        # # Load gripper image tokens
        # gripper_img_dir = osp.join(gripper_vq_dir, scene)
        # if not osp.exists(gripper_img_dir):
        #     continue
        # gripper_img_files = [osp.join(gripper_img_dir, file) for file in sorted(os.listdir(gripper_img_dir), key=sort_by_int)]
                # ------- NEW: Search the current scene across multiple code directories -------

        # ------- NEW: Collect all variants for the current scene (original + augshift, etc.) -------

        # 1) Image tokens: check each directory in vq_dirs
        img_variants = []  # Each entry is either [file1, file2, ...] or None
        for vq_dir in vq_dirs:
            img_dir = osp.join(vq_dir, scene)
            if not osp.exists(img_dir):
                img_variants.append(None)
                continue
            files = [
                osp.join(img_dir, file)
                for file in sorted(os.listdir(img_dir), key=sort_by_int)
            ]
            if len(files) < min_frames:
                img_variants.append(None)
            else:
                img_variants.append(files)

        # 2) Gripper image tokens: check each directory in gripper_vq_dirs
        gripper_variants = []
        for gripper_vq_dir in gripper_vq_dirs:
            gripper_img_dir = osp.join(gripper_vq_dir, scene)
            if not osp.exists(gripper_img_dir):
                gripper_variants.append(None)
                continue
            files = [
                osp.join(gripper_img_dir, file)
                for file in sorted(os.listdir(gripper_img_dir), key=sort_by_int)
            ]
            if len(files) < min_frames:
                gripper_variants.append(None)
            else:
                gripper_variants.append(files)

        # Skip this scene entirely if it is invalid in every variant
        has_valid_variant = False

        # Assume vq_dirs and gripper_vq_dirs are aligned one-to-one (for example:
        #   codes_200           <-> gripper_codes_200
        #   codes_200_augshift  <-> gripper_codes_200_augshift
        # then index-based pairing is sufficient here)
        num_variants = min(len(img_variants), len(gripper_variants))
        for idx in range(num_variants):
            img_files = img_variants[idx]
            gripper_img_files = gripper_variants[idx]
            if img_files is None or gripper_img_files is None:
                continue

            has_valid_variant = True

            # Appending once here corresponds to one variant of this scene (original or augmented)
            result_file.append({
                "text": text,
                "image": img_files,
                "action": action,
                "gripper_image": gripper_img_files,
            })

        # Skip this scene entirely if every variant is invalid
        if not has_valid_variant:
            continue

    print(f"Total number of valid scenes: {len(result_file)}")
    if not result_file:
        raise ValueError("No valid scenes found. Check your dataset path.")

    # === Normalize actions ===
    normalizer = RunningStats()
    action_data = np.concatenate([scene["action"] for scene in result_file])
    normalizer.update(action_data)
    stats = normalizer.get_statistics()

    print("Mean:", stats.mean)
    print("Std:", stats.std)
    print("Q01:", stats.q01)
    print("Q99:", stats.q99)

    for scene in result_file:
        action = scene["action"]
        # Normalize to [-1, 1] using Q01 and Q99 as bounds
        normalized = 2 * (action - stats.q01) / (stats.q99 - stats.q01 + 1e-8) - 1
        scene["action"] = np.clip(normalized, -1, 1)

    # === Save normalized dataset ===
    output_file = osp.join(output_path, output_filename)
    with open(output_file, "wb") as f:
        pickle.dump(result_file, f)
    print(f"Saved normalized data to {output_file}")

    # === Save normalization statistics ===
    save(normalizer_path, {"libero": stats})
    print(f"Saved normalizer statistics to {normalizer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize Libero dataset action values.")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed_data", help="Root path to dataset.")
    parser.add_argument("--output_path", type=str, default="datasets/processed_data/meta", help="Path to save normalized data.")
    parser.add_argument("--normalizer_path", type=str, default="configs/normalizer_libero", help="Path to save normalization stats.")
    parser.add_argument("--output_filename", type=str, default="libero_all_norm.pkl", help="Filename for normalized pickle output.")
    args = parser.parse_args()

    main(args.dataset_path, args.output_path, args.normalizer_path, args.output_filename)

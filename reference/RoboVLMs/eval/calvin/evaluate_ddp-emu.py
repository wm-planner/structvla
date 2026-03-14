"""Evaluate the default decision transformer."""

import argparse
import json
import logging
from pathlib import Path
import time
from collections import Counter, defaultdict, namedtuple
from moviepy.editor import ImageSequenceClip
import copy
from tqdm.auto import tqdm
import sys
import os
import hydra
from omegaconf import OmegaConf
from termcolor import colored

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import torch.multiprocessing as mp

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_env.envs.play_table_env import get_env

from pytorch_lightning import seed_everything
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from robovlms.utils.config_utils import load_config
from model_wrapper import CustomModel
from eval_utils import print_and_save
from model_wrapper_emu import EmuVLAModel
logger = logging.getLogger(__name__)

# Here to adjust the ACT_CHUNK and EP_LEN, our model get better performance with longer EP_LEN
ACT_CHUNK = 10
EP_LEN = 360 // ACT_CHUNK 
NUM_SEQUENCES = 1000

def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    return get_env(val_folder, show_gui=False)

def setup():
    dist.init_process_group(backend="nccl")
    print(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def evaluate_policy(
    rank,
    world_size,
    model,
    env,
    eval_sr_path,
    eval_log_dir=None,
    debug=False,
    raw_calvin=False,
    diverse_inst=False,
):
    """Run this function to evaluate a model on the CALVIN challenge."""
    ### hard code the conf_dir
    conf_dir = Path("/remote-home/jinminghao/OmniSim/reference/RoboVLMs-main/calvin/calvin_models") / "conf"
    ### 

    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    if diverse_inst:
        with open("configs/data/calvin/lang_annotation_cache.json", "r") as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(
            conf_dir / "annotations/new_playtable_validation.yaml"
        )

    eval_log_dir = get_log_dir(eval_log_dir)

    # eval_sequences = get_sequences(NUM_SEQUENCES)
    with open("configs/data/calvin/eval_sequences.json", "r") as f:
        eval_sequences = json.load(f)
    eval_sequences = eval_sequences[:NUM_SEQUENCES]
    eval_sequences = eval_sequences[rank::world_size]
    results = []

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = rank
    local_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            debug,
            eval_log_dir,
            sequence_i,
            raw_calvin,
            diverse_inst=diverse_inst,
        )
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, "a") as f:
                line = f"{local_i}/{NUM_SEQUENCES}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += world_size
                local_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join(
                    [
                        f"{i + 1}/5 : {v * 100:.1f}% |"
                        for i, v in enumerate(success_list)
                    ]
                )
                + "|"
            )
        else:
            sequence_i += world_size
            local_i += 1
    return results


def evaluate_policy_ddp(
    rank,
    world_size,
    model,
    env,
    eval_sr_path,
    eval_log_dir=None,
    debug=False,
    raw_calvin=False,
    create_plan_tsne=False,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    epoch = 0
    conf_dir = Path("path/to/calvin/calvin_models") / "conf"
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml"
    )

    eval_log_dir = get_log_dir(eval_log_dir)

    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    interval_len = int(NUM_SEQUENCES // device_num)

    eval_sequences = get_sequences(NUM_SEQUENCES)
    eval_sequences = eval_sequences[rank::world_size]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            debug,
            eval_log_dir,
            base_sequence_i + local_sequence_i,
            raw_calvin,
        )
        results.append(result)

        success_list = count_success(results)
        with open(eval_sr_path, "a") as f:
            line = f"{local_sequence_i}/{interval_len}: "
            for sr in success_list:
                line += f"{sr:.3f} | "
            line += "\n"
            f.write(line)
        eval_sequences.set_description(
            " ".join(
                [f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]
            )
            + "|"
        )

        local_sequence_i += 1

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    return results


def evaluate_sequence(
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    debug,
    eval_log_dir,
    sequence_i,
    raw_calvin,
    diverse_inst=False,
):
    """Evaluates a sequence of language instructions."""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(
            env,
            model,
            task_checker,
            subtask,
            val_annotations,
            debug,
            eval_log_dir,
            subtask_i,
            sequence_i,
            raw_calvin,
            diverse_inst=diverse_inst,
        )
        # import pdb
        # pdb.set_trace()
        if success:
            success_counter += 1
        else:
            return success_counter

    return success_counter


def rollout(
    env,
    model,
    task_oracle,
    subtask,
    val_annotations,
    debug,
    eval_log_dir,
    subtask_i,
    sequence_i,
    raw_calvin=False,
    diverse_inst=False,
):
    """Run the actual rollout on one subtask (which is one natural language instruction)."""
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
        img_list = []
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    # from time import time
    for _ in range(EP_LEN):
        # time1 = time()
        action = model.step(obs, lang_annotation)
        # time2 = time()
        # print(f"Model step execution time: {time2 - time1:.4f} seconds")
        if action.ndim == 2:
            for single_action in action:
                # time1 = time()
                obs, _, _, current_info = env.step(single_action)
                # time2 = time()
                # print(f"Env step execution time: {time2 - time1:.4f} seconds")
                if debug:
                    img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
                    img_list.append(img_copy)
                # time1 = time()
                # check if current step solves a task
                current_task_info = task_oracle.get_task_info_for_set(
                    start_info, current_info, {subtask}
                )
                # time2 = time()
                # print(f"Task oracle execution time: {time2 - time1:.4f} seconds")
                if len(current_task_info) > 0:
                    if debug:
                        print(colored("success", "green"), end=" ")
                        clip = ImageSequenceClip(img_list, fps=30)
                        clip.write_gif(
                            os.path.join(
                                eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-succ.gif"
                            ),
                            fps=30,
                        )
                    return True
        else:
            obs, _, _, current_info = env.step(action)
            
            if debug:
                img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
                img_list.append(img_copy)

            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(
                start_info, current_info, {subtask}
            )
            if len(current_task_info) > 0:
                if debug:
                    print(colored("success", "green"), end=" ")
                    clip = ImageSequenceClip(img_list, fps=30)
                    clip.write_gif(
                        os.path.join(
                            eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-succ.gif"
                        ),
                        fps=30,
                    )
                return True

    if debug:
        print(colored("fail", "red"), end=" ")
        clip = ImageSequenceClip(img_list, fps=30)
        clip.write_gif(
            os.path.join(eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-fail.gif"),
            fps=30,
        )
    return False


def parser_args():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )
    # default dataset path
    parser.add_argument(
        "--dataset_path",
        default="/remote-home/jinminghao/datasets/robotics/task_ABC_D",
        type=str,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info and visualize environment.",
    )

    parser.add_argument("--emu_hub", type=str, default="")
    parser.add_argument("--vq_hub", type=str, default="/remote-home/jinminghao/pretrain/Emu3-Base")
    parser.add_argument("--vision_hub", type=str, default="/remote-home/jinminghao/structvla/pretrain/Emu3-VisionVQ")
    parser.add_argument("--device_id", default=0, type=int, help="CUDA device")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--raw_calvin", action="store_true")
    parser.add_argument("--debug_model", action="store_true")
    parser.add_argument("--diverse_inst", action="store_true")
    parser.add_argument("--CACHE_ROOT", type=str, default='/remote-home/jinminghao/structvla/logs/calvin_exp_main/structvla_calvin_abcd_video')

    args = parser.parse_args()
    return args

def main():
    args = parser_args()
    setup()
    CACHE_ROOT = args.CACHE_ROOT
    eval_log_dir = os.path.join(CACHE_ROOT, 'eval')
    os.makedirs(eval_log_dir, exist_ok=True)

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    env = make_env(args.dataset_path)


    model = EmuVLAModel(
        emu_hub=args.emu_hub,
        vq_hub=args.vq_hub,
        vision_hub=args.vision_hub,
        device=torch.device("cuda"),
        raw_calvin=args.raw_calvin
    )

    sr_path = os.path.join(eval_log_dir, f"success_rate_calvin.txt")
    result_path = os.path.join(
        eval_log_dir, f"results_calvin_rand-{args.rank}.json"
    )
    cache_file = os.path.join(eval_log_dir, f"meta_info.json")

    if not args.no_cache and args.local_rank == 0:
        if os.path.exists(cache_file):
            os.system(f"sudo rm {cache_file}")
        with open(cache_file, "w") as f:
            _info = {
                "eval_sr_path": sr_path,
                "eval_result_path": result_path,
                "eval_log_dir": eval_log_dir,
            }
            json.dump(_info, f, indent=2)

    results = evaluate_policy(
        args.rank,
        args.world_size,
        model,
        env,
        eval_sr_path=sr_path,
        eval_log_dir=eval_log_dir,
        debug=args.debug,
        diverse_inst=args.diverse_inst,
    )

    # gather results
    with open("configs/data/calvin/eval_sequences.json", "r") as f:
        eval_sequences = json.load(f)
    eval_sequences = eval_sequences[:NUM_SEQUENCES]
    eval_sequences = eval_sequences[args.rank :: args.world_size]
    print_and_save(results, eval_sequences, result_path, None)

    if args.no_cache and args.local_rank == 0:
        os.system("sudo rm -r ./temp/")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

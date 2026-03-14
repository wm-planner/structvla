import os

import numpy as np
import tensorflow as tf
import json

import time
import argparse
#from simpler_env.evaluation.argparse import get_args
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.task_arguments import get_pick_coke_can_args, get_open_drawer_args, get_move_near_args, get_stack_cube_args, get_put_in_drawer_args, get_put_carrot_args, get_put_spoon_args, get_put_eggplant_args
from simpler_env.evaluation.maniskill2_evaluator_batch_acc import run_maniskill2_eval_single_episode
from prismatic.models import load_vla
import multiprocessing as mp 

def get_args():
    parser = argparse.ArgumentParser(description="Description of your script")
    # Add arguments
    parser.add_argument('--POLICY_MODEL', type=str, help='Description of policy_model')
    parser.add_argument('--POLICY_SETUP', type=str, help='Description of policy_setup')
    parser.add_argument('--POLICY_CKPT', type=str, help='Description of policy_ckpt')
    parser.add_argument('--ACTION_SCALE', type=float, help='Description of action_scale')
    parser.add_argument('--ROBOT', type=str, help='Description of robot')
    parser.add_argument('--TF_MEMORY_LIMIT', type=int, help='Description of tf_memory_limit')
    parser.add_argument('--diffusion_depth', type=int,  help='Description of diffusion_depth')
    parser.add_argument('--diffusion_width', type=int, help='Description of diffusion_width')
    parser.add_argument('--cfg_scale', type=float, help='Description of cfg_scale')
    parser.add_argument('--future_action_window_size', type=int, help='Description of future_action_window_size')
    parser.add_argument('--model_type', type=str, help='Description of model_type')
    parser.add_argument('--past_action_window_size', type=int, help='Description of past_action_window_size')
    parser.add_argument('--past_image_window_size', type=int, help='Description of past_image_window_size')
    #parser.add_argument('--additional_env_save_tags', type=str, help='Description of additional_env_save_tags')
    parser.add_argument('--use_ddim', action='store_true', help='Description of use_ddim')
    parser.add_argument('--num_ddim_steps', type=int, help='Description of num_ddim_steps')
    parser.add_argument('--logging_dir', type=str, default=None, help='Description of logging_dir')
    parser.add_argument('--exec_horizon', type=int, default=1, help='Description of exec_horizon')
    parser.add_argument('--use_bf16', action='store_true', help='Enable bf16 usage')  
    parser.add_argument('--action_ensemble',  action='store_true', help='Description of action_ensemble')
    parser.add_argument('--use_ema',  action='store_true', help='Description of use_ema')
    parser.add_argument('--max_num_simulators', type=int, default=0, help='Description of max_num_simulators')
    parser.add_argument('--record_action_horizon', type=int, default=1, help='Description of record_action_horizon')
    parser.add_argument('--action_ensemble_temp', type=float, default=0.2, help='Description of action_ensemble_temp')
    parser.add_argument('--action_ensemble_sim', type=float, default=0.1, help='Description of action_ensemble_sim')
    command_args = parser.parse_args()
    return command_args

# POLICY_MODEL = "openvla"
# POLICY_SETUP = "google_robot"
# #POLICY_CKPT = "/home/v-qixiuli/shared_data/openvla/ft_diffusion_8_d3_w1536/prism-dinosiglip-224px+rt_1+diffusion+n1+b48+x7--image_aug/checkpoints/step-007500-epoch-00-loss=0.0948.pt"
# POLICY_CKPT = "openvla/openvla-7b"
# #POLICY_CKPT = "/home/v-qixiuli/shared_data/openvla/ft_diffusion_8_d3_w1536/prism-dinosiglip-224px+rt_1+diffusion+n1+b48+x7--image_aug/checkpoints/step-007500-epoch-00-loss=0.0948.pt"
# ACTION_SCALE = 1.0
# ROBOT = "google_robot_static"
# TF_MEMORY_LIMIT = 3072

# diffusion_depth = 3
# diffusion_width = 1536
# cfg_scale = 1.5

TRAILS_NUM = {
    "put carrot": 0, # maximum 24
    "stack cube": 24, # maximum 24
    "put spoon": 0, # maximum 24
    "put eggplant": 0, # maximum 24
}   

def init_tasks_args(command_args):
    experiment_args = {}
    np.random.seed(42)
    # set seed for np randint 
    put_carrot_args = get_put_carrot_args()
    put_carrot_idx = np.random.choice(len(put_carrot_args), size=TRAILS_NUM["put carrot"], replace=False)
    experiment_args["put carrot"] = [put_carrot_args[idx] for idx in put_carrot_idx] * 5
    stack_cube_args = get_stack_cube_args()
    stack_cube_idx = np.random.choice(len(stack_cube_args), size=TRAILS_NUM["stack cube"], replace=False)
    experiment_args["stack cube"] = [stack_cube_args[idx] for idx in stack_cube_idx] * 500
    put_spoon_args = get_put_spoon_args()
    put_spoon_idx = np.random.choice(len(put_spoon_args), size=TRAILS_NUM["put spoon"], replace=False)
    experiment_args["put spoon"] = [put_spoon_args[idx] for idx in put_spoon_idx] * 5
    put_eggplant_args = get_put_eggplant_args()
    put_eggplant_idx = np.random.choice(len(put_eggplant_args), size=TRAILS_NUM["put eggplant"], replace=False)
    experiment_args["put eggplant"] = [put_eggplant_args[idx] for idx in put_eggplant_idx] * 5
    return experiment_args

    

if __name__ == "__main__":
    command_args = get_args()

    POLICY_MODEL = command_args.POLICY_MODEL
    #POLICY_CKPT = "/home/v-qixiuli/shared_data/openvla/ft_diffusion_8_d3_w1536/prism-dinosiglip-224px+rt_1+diffusion+n1+b48+x7--image_aug/checkpoints/step-007500-epoch-00-loss=0.0948.pt"
    POLICY_CKPT = command_args.POLICY_CKPT
    #POLICY_CKPT = "/home/v-qixiuli/shared_data/openvla/ft_diffusion_8_d3_w1536/prism-dinosiglip-224px+rt_1+diffusion+n1+b48+x7--image_aug/checkpoints/step-007500-epoch-00-loss=0.0948.pt"
    ACTION_SCALE = command_args.ACTION_SCALE
    TF_MEMORY_LIMIT = command_args.TF_MEMORY_LIMIT

    diffusion_depth = command_args.diffusion_depth
    diffusion_width = command_args.diffusion_width
    cfg_scale = command_args.cfg_scale
    future_action_window_size = command_args.future_action_window_size
    past_action_window_size = command_args.past_action_window_size
    past_image_window_size = command_args.past_image_window_size
    model_type = command_args.model_type
    use_ddim = command_args.use_ddim
    num_ddim_steps = command_args.num_ddim_steps
    logging_dir = command_args.logging_dir
    exec_horizon = command_args.exec_horizon
    use_bf16 = command_args.use_bf16
    action_ensemble = command_args.action_ensemble
    max_num_simulators = command_args.max_num_simulators
    use_ema = command_args.use_ema
    record_action_horizon = command_args.record_action_horizon
    action_ensemble_temp = command_args.action_ensemble_temp
    action_ensemble_sim = command_args.action_ensemble_sim
    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    print("*****************use bf16", use_bf16)
    print("*****************use_ddim", use_ddim)
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=TF_MEMORY_LIMIT)],
        )

    
    #torch.manual_seed(seed)
    if model_type == "mlp":
        model = load_vla(
                    POLICY_CKPT, 
                    load_for_training=False, 
                    load_divia=True,
                    diffusion_depth = diffusion_depth,
                    diffusion_width = diffusion_width,
                    token_size=4096,
                    action_dim=7,
                    future_action_window_size = future_action_window_size,
                    )
    else:
        model = load_vla(
                POLICY_CKPT, 
                load_for_training=False, 
                diffusion_model_type=model_type, 
                token_size=4096, 
                action_dim=7,
                future_action_window_size = future_action_window_size,
                past_action_window_size = past_action_window_size,
                use_ema = use_ema
                )
    if use_ema:
        model.load_ema_to_weights()
    
    model.to("cuda")
    model.eval()



    # elif POLICY_MODEL == "DiTVLA_ddim":
    #     from simpler_env.policies.diffusionvla.ditvla_ddim import DiTVLA_DDIM_Inference
    #     assert POLICY_CKPT is not None
    #     model = DiTVLA_DDIM_Inference(
    #         saved_model_path=POLICY_CKPT,
    #         policy_setup=POLICY_SETUP,
    #         model_type = model_type,
    #         future_action_window_size = future_action_window_size,
    #         past_action_window_size = past_action_window_size,
    #         past_image_window_size = past_image_window_size,
    #         cfg_scale = cfg_scale,
    #         action_scale=ACTION_SCALE,
    #         use_ddim=use_ddim,
    #         num_ddim_steps=num_ddim_steps,
    #     )    
    # else:
    #     raise NotImplementedError()
    
    success_arr = {}

    experiment_args = init_tasks_args(command_args)
    start_time = time.time()

    cur_num_simulators = 0
    for task in experiment_args.keys():
        cur_num_simulators = 0
        success_arr[task] = []
        len(experiment_args[task]) % max_num_simulators
        for cur_num, args in enumerate(experiment_args[task]):
            # initialize the environment args list
            if (cur_num) % max_num_simulators == 0 or cur_num == 0:
                robot_name_list = []
                env_name_list = []
                scene_name_list = []
                robot_init_x_list = []
                robot_init_y_list = []
                robot_init_quat_list = []
                control_mode_list = []
                additional_env_build_kwargs_list = []
                rgb_overlay_path_list = []
                control_freq_list = []
                sim_freq_list = []
                max_episode_steps_list = []
                enable_raytracing_list = []
                additional_env_save_tags_list = []
                obs_camera_name_list = []
                policy_setup_list = []
                obj_init_x_list = []
                obj_init_y_list = []
                obj_episode_id_list = []
                logging_dir_list = []
            # update the environment args list
            ROBOT = args.robot
            policy_setup = args.policy_setup
            #action_scale 
            control_mode = get_robot_control_mode(ROBOT, POLICY_MODEL)
            robot_name_list.append(ROBOT)
            env_name_list.append(args.env_name)
            scene_name_list.append(args.scene_name)
            robot_init_x_list.append(args.robot_init_x)
            robot_init_y_list.append(args.robot_init_y)
            robot_init_quat_list.append(args.robot_init_quats[0])
            control_mode_list.append(control_mode)
            additional_env_build_kwargs_list.append(args.additional_env_build_kwargs)
            rgb_overlay_path_list.append(args.rgb_overlay_path)
            control_freq_list.append(args.control_freq)
            sim_freq_list.append(args.sim_freq)
            max_episode_steps_list.append(args.max_episode_steps)
            enable_raytracing_list.append(args.enable_raytracing)
            additional_env_save_tags_list.append(args.additional_env_save_tags)
            obs_camera_name_list.append(args.obs_camera_name)
            policy_setup_list.append(policy_setup)
            if args.obj_variation_mode == "xy":
                obj_init_x_list.append(args.obj_init_x)
                obj_init_y_list.append(args.obj_init_y)
                obj_episode_id_list.append(None)
            elif args.obj_variation_mode == "episode":
                obj_episode_id_list.append(args.obj_episode_id)
                obj_init_x_list.append(None)
                obj_init_y_list.append(None)
            if logging_dir is not None:
                logging_dir = logging_dir
            else:
                logging_dir = args.logging_dir
            logging_dir_list.append(logging_dir)
            print(args)
            if (cur_num + 1) % max_num_simulators == 0 or cur_num == len(experiment_args[task]) - 1:
                if (cur_num + 1) % max_num_simulators == 0:
                    cur_num_simulators = max_num_simulators
                else:
                    cur_num_simulators = len(experiment_args[task]) % max_num_simulators
                kwargs = dict(
                    model = model,
                    ckpt_path = POLICY_CKPT,
                    robot_name_list = robot_name_list,
                    env_name_list = env_name_list,
                    scene_name_list = scene_name_list,
                    robot_init_x_list = robot_init_x_list,
                    robot_init_y_list = robot_init_y_list,
                    robot_init_quat_list = robot_init_quat_list,
                    control_mode_list = control_mode_list,
                    additional_env_build_kwargs_list = additional_env_build_kwargs_list,
                    rgb_overlay_path_list = rgb_overlay_path_list,
                    control_freq_list = control_freq_list,
                    sim_freq_list = sim_freq_list,
                    max_episode_steps_list = max_episode_steps_list,
                    enable_raytracing_list = enable_raytracing_list,
                    additional_env_save_tags_list = additional_env_save_tags_list,
                    obs_camera_name_list = obs_camera_name_list,
                    logging_dir_list = logging_dir_list,
                    num_simulators = cur_num_simulators,
                    cfg_scale = cfg_scale,
                    use_ddim = use_ddim,
                    num_ddim_steps = num_ddim_steps,
                    policy_setup_list = policy_setup_list,
                    obj_init_x_list = obj_init_x_list,
                    obj_init_y_list = obj_init_y_list,
                    obj_episode_id_list = obj_episode_id_list,
                    exec_horizon = exec_horizon,
                    action_ensemble = action_ensemble,
                    record_action_horizon = record_action_horizon,
                    action_ensemble_temp = action_ensemble_temp,
                    action_ensemble_sim = action_ensemble_sim,
                )
                suc = run_maniskill2_eval_single_episode(
                    **kwargs,
                )
                if isinstance(suc, list):
                    success_arr[task].extend(suc)
                else:
                    success_arr[task].append(suc)
            
    for task in success_arr.keys():
        print(f"Task: {task}")
        print(f"Average success: {np.mean(success_arr[task])}; Success trails: {np.count_nonzero(success_arr[task])}/{len(success_arr[task])}")
        print("\n")
    print(f"Total time: {time.time() - start_time}")

    ckpt_path_basename = POLICY_CKPT if POLICY_CKPT[-1] != "/" else POLICY_CKPT[:-1]
    ckpt_path_foldername = ckpt_path_basename.split("/")[-4]
    ckpt_path_modelname = ckpt_path_basename.split("/")[-1]
    trial_string = "_".join([f"{task}{len(experiment_args[task])}" for task in experiment_args.keys()])
    save_folder = os.path.join(logging_dir, ckpt_path_foldername, ckpt_path_modelname)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_file = os.path.join(save_folder, f"{trial_string}.txt")
    total_success_num = 0
    total_exp_num = 0
    with open(save_file, "a") as f:
        f.write(f"{args}\n")
        for task in success_arr.keys():
            f.write(f"Task: {task}\n")
            success_num = np.count_nonzero(success_arr[task])
            total_success_num += success_num
            total_exp_num += len(success_arr[task])
            f.write(f"Average success: {np.mean(success_arr[task])}; Success trails: {success_num}/{len(success_arr[task])}\n")
            f.write("\n")
        f.write(f"Overall success: {total_success_num/total_exp_num}; Success trails: {total_success_num}/{total_exp_num}\n")
        f.write("\n")
        f.write(f"Total time: {time.time() - start_time}\n")
        f.write("\n")
        f.write(json.dumps(success_arr))
        f.write("\n")
        for task in experiment_args.keys():
            for args in experiment_args[task]:
                f.write(f"{args}\n")

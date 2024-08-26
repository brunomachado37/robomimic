import os
import json
import argparse
import torch
import numpy as np

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.lang_utils as LangUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.file_utils import maybe_dict_from_checkpoint


def env_iterator(config, eval_env_meta_list, eval_shape_meta_list):
    for (env_meta, shape_meta) in zip(eval_env_meta_list, eval_shape_meta_list):
        def create_env_helper(env_i=0):
            env_kwargs = dict(
                env_meta=env_meta,
                env_name=env_meta['env_name'],
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
                seed=config.train.seed * 1000 + env_i,
            )
            env = EnvUtils.create_env_from_metadata(**env_kwargs)
            # handle environment wrappers
            env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable

            return env

        if config.experiment.rollout.batched:
            from tianshou.env import SubprocVectorEnv
            env_fns = [lambda env_i=i: create_env_helper(env_i) for i in range(config.experiment.rollout.num_batch_envs)]
            env = SubprocVectorEnv(env_fns)
            # env_name = env.get_env_attr(key="name", id=0)[0]
        else:
            env = create_env_helper()
            # env_name = env.name
        print(env)
        yield env


def eval(args):
    ext_cfg = json.load(open(os.path.join(args.model_path, "config.json")))
    config = config_factory(ext_cfg["algo_name"])

    with config.values_unlocked():
        config.update(ext_cfg)

    ObsUtils.initialize_obs_utils_with_config(config)

    env_config = json.load(open(os.path.join(args.model_path, "env_config.json")))
    env_config['action_normalization_stats'] = {key: {k: np.array(v) for k, v in env_config['action_normalization_stats'][key].items()} for key in env_config['action_normalization_stats']}
    device = torch.device("cuda:0")

    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=env_config["shape_meta_list"][0]["all_shapes"],
        ac_dim=env_config["shape_meta_list"][0]["ac_dim"],
        device=device,
    )

    weights_path = os.path.join(args.model_path, f"models/model_epoch_{args.checkpoint_epoch}.pth")
    
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=weights_path)
    model.deserialize(ckpt_dict["model"])

    lang_encoder = LangUtils.language_encoder_factory(
        model=config.train.language_encoder,
        device=device,
    )

    rollout_model = RolloutPolicy(
        model,
        obs_normalization_stats=env_config['obs_normalization_stats'],
        action_normalization_stats=env_config['action_normalization_stats'],
        lang_encoder=lang_encoder,
    )

    video_path = os.path.join(args.output_dir, config.experiment.name, "videos")
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
        policy=rollout_model,
        envs=env_iterator(config, env_config['env_meta_list'], env_config['shape_meta_list']),
        horizon=[env['horizon'] for env in config.train.data],
        use_goals=config.use_goals,
        num_episodes=args.evaluations_per_task,
        render=False,
        video_dir=video_path,
        epoch=args.checkpoint_epoch,
        video_skip=config.experiment.get("video_skip", 5),
        terminate_on_success=config.experiment.rollout.terminate_on_success,
        del_envs_after_rollouts=True,
        data_logger=None,
    )

    for task, ret in all_rollout_logs.items():
        print(f"\n{task} success rate: {ret['Success_Rate']:.2f}\n")

    all_rollout_logs["rollouts_per_task"] = args.evaluations_per_task
    eval_results_path = os.path.join(args.output_dir, config.experiment.name, f"eval_results_epoch_{args.checkpoint_epoch}.json")

    with open(eval_results_path, "w") as f:
        json.dump(all_rollout_logs, f, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default='/home/liris/bmachado/expdata/robocasa/im/bc_xfmr/08-05-None/seed_123_ds_mg-3000/20240814202051',
        help="Path to the run foler containing the model to evaluate",
    )

    parser.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=30,
        help="Epoch of the checkpoint to evaluate",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='eval',
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--evaluations_per_task",
        type=int,
        default=3,
        help="Number of rollouts to perform per task",
    )

    args = parser.parse_args()
    eval(args)

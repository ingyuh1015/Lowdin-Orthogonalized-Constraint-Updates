import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import asdict

import bullet_safety_gym

import gymnasium as gym
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from fsrl.agent import TRPOLagAgent
from fsrl.config.trpol_cfg import (

    BC_v2,
    CC_v2,
    DC_v2,
    BC_v3,
    CC_v3,
    DC_v3,
    TrainCfg,
)

from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, select_redundant_wrapper

TASK_TO_CFG = {
    # bullet safety gym tasks
    "BC-v2": BC_v2,
    "CC-v2": CC_v2,
    "DC-v2": DC_v2,
    "BC-v3": BC_v3,
    "CC-v3": CC_v3,
    "DC-v3": DC_v3,
}


@pyrallis.wrap()
def train(args: TrainCfg):

    task = args.task
    short_task = args.short_task

    default_cfg = TASK_TO_CFG[short_task]() if short_task in TASK_TO_CFG else TrainCfg()
    args.device = "cuda:0"
    # use the default configs instead of the input args.
    if args.use_default_cfg:
        base_cfg = TrainCfg()
        cfg_dict = asdict(default_cfg)
        args_dict = asdict(args)
        base_dict = asdict(base_cfg)
        for key, val in args_dict.items():
            if val is not None and val != base_dict.get(key):
                cfg_dict[key] = val
        for key, val in cfg_dict.items():
            setattr(default_cfg, key, val)
        args = default_cfg

    cost_0_thres = args.thre_0
    cost_1_thres = args.thre_1
    cost_2_thres = args.thre_2

    # cost_limit_new = [cost_0_thres, cost_1_thres, cost_2_thres]
    cost_limit_new = []

    args.cost_0_redundant_num = redundant_num
    args.cost_1_redundant_num = redundant_num
    args.cost_2_redundant_num = redundant_num


    for _ in range(args.cost_0_redundant_num):
        cost_limit_new.append(cost_0_thres)
    for _ in range(args.cost_1_redundant_num):
        cost_limit_new.append(cost_1_thres)
    for _ in range(args.cost_2_redundant_num):
        cost_limit_new.append(cost_2_thres)

    args.cost_limit = cost_limit_new
    # print(args.cost_limit)
    # raise RuntimeError

    # setup logger
    cfg = asdict(args)
    default_cfg = asdict(default_cfg)
    if args.name is None:
        args.name = "GradS(TRPOL)-" + auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = "GradS-" + args.short_task # + "-cost-0-" + str(cost_0_thres) + "-cost-1-" + str(cost_1_thres) + "-cost-2-" + str(cost_2_thres) # + \
        if args.cost_0_redundant_num > 0:
            args.group += ("-cost-0-" + str(cost_0_thres))
        if args.cost_1_redundant_num > 0:
            args.group += ("-cost-1-" + str(cost_1_thres))
        if args.cost_2_redundant_num > 0:
            args.group += ("-cost-2-" + str(cost_1_thres))
        if args.prefix:
            args.group += "-" + str(args.prefix)
        group_mode = "-vanilla" if args.vanilla else "-grads_true"
        args.group += group_mode
        args.group += args.suffix


    redundant_num = max(args.cost_0_redundant_num, args.cost_1_redundant_num, args.cost_2_redundant_num)
    redundant_wrapper_cls = select_redundant_wrapper(redundant_num)
    args.project = f"{args.short_task}-redun-{redundant_num}"
    args.logdir = os.path.join("logs", args.project, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir) # BaseLogger() #
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)



    demo_env = gym.make(args.task)

    agent = TRPOLagAgent(
        env=demo_env,
        logger=logger,
        device="cuda:0",
        thread=args.thread,
        seed=args.seed,
        lr=args.lr,
        hidden_sizes=args.hidden_sizes,
        unbounded=args.unbounded,
        last_layer_scale=args.last_layer_scale,
        target_kl=args.target_kl,
        backtrack_coeff=args.backtrack_coeff,
        max_backtracks=args.max_backtracks,
        optim_critic_iters=args.optim_critic_iters,
        gae_lambda=args.gae_lambda,
        advantage_normalization=args.norm_adv,
        use_lagrangian=args.use_lagrangian,
        lagrangian_pid=args.lagrangian_pid,
        cost_limit=args.cost_limit,
        rescaling=args.rescaling,
        gamma=args.gamma,
        max_batchsize=args.max_batchsize,
        reward_normalization=args.rew_norm,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        vanilla=args.vanilla,
        random_con_selection=args.random_con_selection,
        gradient_shaping=args.gradient_shaping,
        max_singleAC = args.max_singleAC,
        conditioned_sigma = args.conditioned_sigma
    )

    training_num = min(args.training_num, args.episode_per_collect)
    worker = eval(args.worker)


    train_envs = worker([lambda: redundant_wrapper_cls(gym.make(args.task), 
                                                       cost_0_redunt_num=args.cost_0_redundant_num, 
                                                       cost_1_redunt_num=args.cost_1_redundant_num,
                                                       cost_2_redunt_num=args.cost_2_redundant_num) for _ in range(training_num)])
        

    test_envs = None
    valid_cost_num = len(cost_limit_new)
    
    # start training
    agent.learn(
        train_envs=train_envs,
        test_envs=test_envs,
        epoch=args.epoch,
        episode_per_collect=args.episode_per_collect,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        buffer_size=args.step_per_epoch,
        testing_num=args.testing_num,
        batch_size=args.batch_size,
        reward_threshold=args.reward_threshold,
        save_interval=args.save_interval,
        resume=args.resume,
        save_ckpt=args.save_ckpt,
        verbose=args.verbose,
        valid_cost_num=valid_cost_num
    )

if __name__ == "__main__":
    train()

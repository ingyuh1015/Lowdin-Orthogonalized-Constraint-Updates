import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')

    import bullet_safety_gym
    import gymnasium as gym
    import torch
    import yaml
    from fsrl.agent import LOCUAgent
    from fsrl.utils.exp_util import select_redundant_wrapper

    with open(args.config) as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    counts = [int(cfg[f'cost_{i}_redundant_num']) for i in range(3)]
    limits = [cfg[f'thre_{i}'] for i in range(3) for _ in range(counts[i])]
    wrapper = select_redundant_wrapper(max(counts))
    env = wrapper(gym.make(cfg['task']), **{
        f'cost_{i}_redunt_num': counts[i] for i in range(3)
    })
    try:
        aliases = {'advantage_normalization': 'norm_adv', 'reward_normalization': 'rew_norm'}
        kwargs = {
            name: cfg[aliases.get(name, name)]
            for name in inspect.signature(LOCUAgent.__init__).parameters
            if aliases.get(name, name) in cfg
            and name not in ('self', 'env', 'logger', 'device', 'cost_limit', 'constraint_num')
        }
        agent = LOCUAgent(env=env, device=args.device, cost_limit=limits,
                          constraint_num=len(limits), **kwargs)
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        reward, length, cost = agent.evaluate(
            env, state_dict=checkpoint['model'], eval_episodes=args.episodes
        )
        print(f'Reward: {reward}\nEpisode length: {length}\nCost: {cost}')
    finally:
        env.close()


if __name__ == '__main__':
    main()

import sys
import os
sys.path.append(os.path.relpath("."))
import argparse
from enviroments.flocking_enviroment import *
from models.FFNC import *
from MADDPG import *
from SharedMADDPG import SharedMADDPG
from sac import SAC
import torch


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=40)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--memory_capacity", type=int, default=1000000, help="Maximum number of memory replays to store.")
    parser.add_argument("--pre_train_eps", type=int, default=10, help="Number of episodes to run before starting training. Allows memory to be built up.")
    # Model arguments
    args = parser.parse_args()

    env = FlockEnviroment(args.num_agents)

    # algo = MADDPG(args.num_agents, env.get_observation_size(), env.dimensions, args.batch_size,
    #              args.memory_capacity, args.pre_train_eps)

    
    # algo = SharedMADDPG(args.num_agents, env.get_observation_size(), env.dimensions, args.batch_size,
    #                     args.memory_capacity, args.pre_train_eps)
    
    kwargs = {"policy": "Gaussian", "gpuid": 2} 
    algo = SAC(args.num_agents, env.get_observation_size(), np.array([[0.25, 0.25],[-0.25, -0.25]]), **kwargs)
    
    last_10_collisions = np.zeros(10)
    while (algo.episode_done < args.num_episodes):
        print(algo.episode_done, "/", args.num_episodes)
        state = env.reset()
        done = False
        total_reward = 0.0
        while(not done):
            actions = algo.select_action(algo.to_float_tensor(state))
            old_state, rewards, state, done = env.step(actions.clone().detach().cpu().numpy())
            total_reward += rewards.sum()
            # algo.memory.push(old_state, actions,\
            #     state, rewards)
            mask = float(not done)
            algo.memory.push(old_state, actions, rewards, state, mask)

        algo.episode_done += 1
        algo.update_policy()
        # if (algo.episode_done % 10 == 0):
        #     env.display_last_episode()
        print("Collisions", env.episode_collisions)
        print("Total reward", total_reward)
        last_10_collisions[algo.episode_done % 10 - 1] = env.episode_collisions
        if (algo.episode_done % 10 == 0):
            print("--- Last Ten Average", last_10_collisions.mean())
            last_10_collisions = np.zeros(10)

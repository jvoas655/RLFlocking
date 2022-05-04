import sys
import os
import numpy as np
sys.path.append(os.path.relpath("."))
import argparse
from enviroments.flocking_enviroment import *
from models.FFNC import *
from MADDPG import *
from PS_CACER import *
import torch

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=40)
    parser.add_argument("--num_episodes", type=int, default=10000000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--memory_capacity", type=int, default=800000, help="Maximum number of memory replays to store.")
    parser.add_argument("--pre_train_eps", type=int, default=5, help="Number of episodes to run before starting training. Allows memory to be built up.")
    parser.add_argument("--eval_frequency", type=int, default=10, help="")
    parser.add_argument("--eval_episodes", type=int, default=3, help="")
    # Model arguments
    args = parser.parse_args()

    env = FlockEnviroment(args.num_agents)

    algo = PS_CACER(args.num_agents, env.get_observation_size(), env.dimensions, args.batch_size,
                 args.memory_capacity, args.pre_train_eps)
    last_10_collisions = np.zeros(10)
    all_evals = []
    while (algo.episode_done < args.num_episodes):
        print(algo.episode_done, "/", args.num_episodes)
        state = env.reset()
        done = False
        while(not done):
            actions = algo.select_action(algo.to_float_tensor(state))
            np_actions = np.array(list(map(lambda t: t.clone().detach().cpu().numpy(), actions)))
            old_state, rewards, state, done = env.step(np_actions)
            algo.memory.push(args.num_agents, old_state, actions, state, rewards)
        algo.update_policy()
        print(len(algo.memory), "/", args.memory_capacity)
        #if (algo.episode_done % 100 == 0):
            #env.display_last_episode()
        if (algo.episode_done % args.eval_frequency == 0):
            eval_store = []
            for eval in range(args.eval_episodes):
                state = env.reset()
                done = False
                epi_store = []
                while(not done):
                    actions = algo.select_action(algo.to_float_tensor(state), noise=False)
                    np_actions = np.array(list(map(lambda t: t.clone().detach().cpu().numpy(), actions)))
                    old_state, rewards, state, done = env.step(np_actions)
                    epi_store.append(np.sum(rewards))
                eval_store.append(np.mean(epi_store))
            mean_r = np.mean(eval_store)
            print("Eval R:", mean_r)
            all_evals.append(mean_r)
        algo.episode_done += 1

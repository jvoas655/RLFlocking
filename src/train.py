import sys
import os
import numpy as np
sys.path.append(os.path.relpath("."))
import argparse
from enviroments.flocking_enviroment import *
from models.FFNC import *
from PS_CACER import *
from DDPG import *
from memory import *
import torch

if __name__ == "__main__":
    #torch.random.manual_seed(0)
    #np.random.seed(0)
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=40)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--memory_capacity", type=int, default=800000, help="Maximum number of memory replays to store.")
    parser.add_argument("--pre_train_eps", type=int, default=1, help="Number of episodes to run before starting training. Allows memory to be built up.")
    parser.add_argument("--eval_frequency", type=int, default=10, help="")
    parser.add_argument("--eval_episodes", type=int, default=3, help="")
    parser.add_argument("--use_memory", action="store_true", default=False, help="")
    # Model arguments
    args = parser.parse_args()

    env = FlockEnviroment(args.num_agents)

    algo = DDPG(args.num_agents, env.get_observation_size(), env.dimensions, args.batch_size,
                 args.memory_capacity, args.pre_train_eps)
    last_10_collisions = np.zeros(10)
    while (algo.episode_done < args.num_episodes):
        print("Episode:", algo.episode_done, "/", args.num_episodes)
        state = env.reset()
        done = False
        exps = []
        while(not done):
            actions = algo.select_action(algo.to_float_tensor(state))
            np_actions = np.array(list(map(lambda t: t.clone().detach().cpu().numpy(), actions)))
            old_state, rewards, state, done = env.step(np_actions)
            if (args.use_memory):
                algo.memory.push(args.num_agents, old_state, actions, state, rewards)
            else:
                exps += list(map(lambda e: Experience(*e), list(zip(old_state, actions, state, rewards))))
        for i in range(10):
            if (args.use_memory):
                algo.update_policy()
                print("---", "Memory:",len(algo.memory), "/", args.memory_capacity)
            else:
                algo.update_policy(exps)
        if (algo.episode_done % args.eval_frequency == 0):
            eval_store = []
            collisions_store = []
            deviation_store = []
            square_deviation_store = []
            for eval in range(args.eval_episodes):
                state = env.reset()
                done = False
                epi_store = []
                collisions_store2 = []
                deviation_store2 = []
                square_deviation_store2 = []
                while(not done):
                    actions = algo.select_action(algo.to_float_tensor(state), noise=False)
                    np_actions = np.array(list(map(lambda t: t.clone().detach().cpu().numpy(), actions)))
                    old_state, rewards, state, done = env.step(np_actions)
                    epi_store.append(np.mean(rewards))
                eval_store.append(np.sum(epi_store))
                collisions_store.append(env.episode_collisions)
                deviation_store.append(env.deviation)
                square_deviation_store.append(env.square_deviation)
            mean_r = np.mean(eval_store)
            mean_c = np.mean(collisions_store)
            mean_d = np.mean(deviation_store)
            mean_d2 = np.mean(square_deviation_store)
            print("Eval R:", mean_r)
            print("Eval C:", mean_c / args.num_agents)
            print("Eval D:", mean_d / args.num_agents)
            print("Eval D2:", mean_d2 / args.num_agents)
            env.display_last_episode("fig" + str(algo.episode_done) + ".jif")
        algo.episode_done += 1

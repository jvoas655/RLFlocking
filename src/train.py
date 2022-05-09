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
import shutil

if __name__ == "__main__":
    #torch.random.manual_seed(0)
    #np.random.seed(0)
    # Arguments
    test_string = "test"  # Change me!
    dir_path = "..\\results\\" + "_".join(test_string.lower().split())
    if (os.path.exists(dir_path)):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=40)
    parser.add_argument("--num_episodes", type=int, default=251)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--memory_capacity", type=int, default=800000, help="Maximum number of memory replays to store.")
    parser.add_argument("--pre_train_eps", type=int, default=0, help="Number of episodes to run before starting training. Allows memory to be built up.")
    parser.add_argument("--eval_frequency", type=int, default=10, help="")
    parser.add_argument("--eval_episodes", type=int, default=5, help="")
    parser.add_argument("--hidden_size", type=int, default=64, help="")
    parser.add_argument("--use_memory", action="store_true", default=False, help="")
    # Model arguments
    args = parser.parse_args()
    env = FlockEnviroment(args.num_agents)

    algo = DDPG(args.num_agents, env.get_observation_size(), env.dimensions, args.batch_size,
                 args.memory_capacity, args.pre_train_eps, args.hidden_size)

    env.center_reward_scale = 1
    env.collision_reward_scale = 0
    training_data_inds = {"Rewards":0, "Collisions":1, "Density":2, "Density2":3, "Critic":4, "Actor":5}
    training_data = np.zeros((len(training_data_inds), args.num_episodes))

    eval_data_inds = {"Collisions":0, "Density":1, "Density2":2}
    eval_data = np.zeros((len(eval_data_inds), int(args.num_episodes / args.eval_frequency) + 1))
    while (algo.episode_done < args.num_episodes):
        print("Episode:", algo.episode_done, "/", args.num_episodes)
        if (algo.episode_done == 50):
            env.center_reward_scale = 0.0
            env.collision_reward_scale = 1.0
            if (args.use_memory):
                algo.memory.clear()
        state = env.reset()
        done = False
        reward_acum = []
        collision_acum = []
        density_acum = []
        exps = []
        while(not done):
            actions = algo.select_action(algo.to_float_tensor(state))
            np_actions = np.array(list(map(lambda t: t.clone().detach().cpu().numpy(), actions)))
            old_state, rewards, state, done = env.step(np_actions)
            reward_acum.append(np.mean(rewards))
            if (args.use_memory):
                algo.memory.push(args.num_agents, old_state, actions, state, rewards)
            else:
                exps += list(map(lambda e: Experience(*e), list(zip(old_state, actions, state, rewards))))
        training_data[training_data_inds["Rewards"], algo.episode_done] = np.sum(reward_acum)
        training_data[training_data_inds["Collisions"], algo.episode_done] = env.episode_collisions
        training_data[training_data_inds["Density"], algo.episode_done] = env.deviation
        training_data[training_data_inds["Density2"], algo.episode_done] = env.square_deviation
        c_acum, a_acum = [], []
        for i in range(10):
            if (args.use_memory):
                c, a = algo.update_policy()
                c_acum.append(c)
                a_acum.append(a)
            else:
                c, a = algo.update_policy(exps)
                c_acum.append(c)
                a_acum.append(a)
        training_data[training_data_inds["Critic"], algo.episode_done] = np.mean(c_acum)
        training_data[training_data_inds["Actor"], algo.episode_done] = np.mean(a_acum)
        if (algo.episode_done % args.eval_frequency == 0):
            collisions_store = []
            deviation_store = []
            square_deviation_store = []
            for eval in range(args.eval_episodes):
                state = env.reset()
                done = False
                collisions_store2 = []
                deviation_store2 = []
                square_deviation_store2 = []
                while(not done):
                    actions = algo.select_action(algo.to_float_tensor(state), noise=False)
                    np_actions = np.array(list(map(lambda t: t.clone().detach().cpu().numpy(), actions)))
                    old_state, rewards, state, done = env.step(np_actions)
                collisions_store.append(env.episode_collisions)
                deviation_store.append(env.deviation)
                square_deviation_store.append(env.square_deviation)
            mean_c = np.mean(collisions_store) / args.num_agents
            mean_d = np.mean(deviation_store) / args.num_agents
            mean_d2 = np.mean(square_deviation_store) / args.num_agents
            eval_data[eval_data_inds["Collisions"], int(algo.episode_done / args.eval_frequency)] = mean_c
            eval_data[eval_data_inds["Density"], int(algo.episode_done / args.eval_frequency)] = mean_d
            eval_data[eval_data_inds["Density2"], int(algo.episode_done / args.eval_frequency)] = mean_d2
            print("Eval C:", mean_c)
            print("Eval D:", mean_d)
            print("Eval D2:", mean_d2)
            env.display_last_episode(test_string + ": " + str(algo.episode_done), dir_path + "\\fig" + str(algo.episode_done) + ".gif")
        algo.episode_done += 1
    for key in training_data_inds  :
        fig, axes = plt.subplots(1, 1)
        #fig.tight_layout()
        axes.set_title("Training: " + key)
        axes.set_xlabel("Episode")
        axes.plot(training_data[training_data_inds[key]])
        plt.savefig(dir_path + "\\" + key + ".png")
        plt.close(fig)
    with open(dir_path + "\\training_keys.txt", "w") as fref:
        keys = [None] * len(training_data_inds)
        for key in training_data_inds:
            keys[training_data_inds[key]] = key
        fref.write("\n".join(keys))
    np.save(dir_path + "\\" + "training_data.npy", training_data)
    for key in eval_data_inds:
        fig, axes = plt.subplots(1, 1)
        #fig.tight_layout()
        axes.set_title("Evaluation: " + key)
        axes.set_xlabel("Episode")
        axes.plot(range(0, args.num_episodes + 1, args.eval_frequency), eval_data[eval_data_inds[key]])
        plt.savefig(dir_path + "\\" + key + ".png")
        plt.close(fig)
    with open(dir_path + "\\eval_keys.txt", "w") as fref:
        keys = [None] * len(eval_data_inds)
        for key in eval_data_inds:
            keys[eval_data_inds[key]] = key
        fref.write("\n".join(keys))
    np.save(dir_path + "\\" + "eval_data.npy", eval_data)

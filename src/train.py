import sys
import os
sys.path.append(os.path.relpath("."))
import argparse
from enviroments.flocking_enviroment import *
from models.FFNC import *
from MADDPG import *

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

    algo = MADDPG(args.num_agents, env.get_observation_size(), env.dimensions, args.batch_size,
                 args.memory_capacity, args.pre_train_eps)
    last_10_collisions = np.zeros(10)
    while (algo.episode_done < args.num_episodes):
        print(algo.episode_done, "/", args.num_episodes)
        state = env.reset()
        done = False
        while(not done):
            actions = algo.select_action(algo.to_float_tensor(state))
            old_state, rewards, state, done = env.step(actions.clone().detach().cpu().numpy())
            algo.memory.push(old_state, actions,\
                state, rewards)
        algo.episode_done += 1
        algo.update_policy()
        #if (algo.episode_done % 10 == 0):
            #env.display_last_episode()
        print("Collisions", env.episode_collisions)
        last_10_collisions[algo.episode_done % 10 - 1] = env.episode_collisions
        if (algo.episode_done % 10 == 0):
            print("--- Last Ten Average", last_10_collisions.mean())
            last_10_collisions = np.zeros(10)

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
    parser.add_argument("--num_agents", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--memory_capacity", type=int, default=10000, help="Maximum number of memory replays to store.")
    parser.add_argument("--pre_train_eps", type=int, default=100, help="Number of episodes to run before starting training. Allows memory to be built up.")
    # Model arguments
    args = parser.parse_args()

    env = FlockEnviroment(args.num_agents)

    algo = MADDPG(args.num_agents, env.get_observation_size(), env.dimensions, args.batch_size,
                 args.memory_capacity, args.pre_train_eps)

    state = env.reset()
    done = False
    while(not done):
        actions = algo.select_action(state)
        _, _, state, done = env.step(actions.detach().cpu().numpy())
    env.display_last_episode()

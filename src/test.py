import sys
import os
sys.path.append(os.path.relpath("."))
import argparse
from enviroments.flocking_enviroment import *
from sac import SAC
import torch



if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=40)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--gpuid", type=int, default=0, help="Which gpu to train on")
    parser.add_argument("--ckpt_path", type=str, default="", help="Path to the loaded checkpoint")
    parser.add_argument("--policy", type=str, default="Deterministic")
    
    args = parser.parse_args()

    env = FlockEnviroment(args.num_agents)

    # algo = MADDPG(args.num_agents, env.get_observation_size(), env.dimensions, args.batch_size,
    #              args.memory_capacity, args.pre_train_eps)

    
    # algo = SharedMADDPG(args.num_agents, env.get_observation_size(), env.dimensions, args.batch_size,
    #                     args.memory_capacity, args.pre_train_eps)


    kwargs = {"policy": args.policy, "gpuid": args.gpuid} 
    algo = SAC(args.num_agents, env.get_observation_size(), np.array([[0.25, 0.25],[-0.25, -0.25]]), **kwargs)
    algo.load_checkpoint(args.ckpt_path)
    
    reward_hist = []
    collision_hist = []
    while (algo.episode_done < args.num_episodes):
        print(algo.episode_done, "/", args.num_episodes)
        state = env.reset()
        done = False
        total_reward = 0.0
        
        while(not done):
            # actions = algo.select_action(algo.to_float_tensor(state))
            # actions = (0.25-(-0.25))*torch.rand((args.num_agents, 2)) + (-0.25) # uniformly random
            mean = torch.zeros((args.num_agents, 2))
            std = torch.ones((args.num_agents, 2)) * 0.1
            actions = torch.normal(mean, std) # Gaussian random
            old_state, rewards, state, done = env.step(actions.clone().detach().cpu().numpy())
            total_reward += rewards.sum()
            # algo.memory.push(old_state, actions,\
            #     state, rewards)
        algo.episode_done += 1
        print("Collisions", env.episode_collisions)
        print("Total reward", total_reward)
        reward_hist.append(total_reward)
        collision_hist.append(env.episode_collisions)
        
        # env.display_last_episode()


    # print("reward history", reward_hist)
    # print("collision history", collision_hist)
    print("average collision cnt", sum(collision_hist)/len(collision_hist))

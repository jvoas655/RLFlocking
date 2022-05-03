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
from torch.utils.tensorboard import SummaryWriter
import datetime



if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=40)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--memory_capacity", type=int, default=1000000, help="Maximum number of memory replays to store.")
    parser.add_argument("--pre_train_eps", type=int, default=10, help="Number of episodes to run before starting training. Allows memory to be built up.")
    parser.add_argument("--update_cycle", type=int, default=16, help="Update every how many steps?")
    parser.add_argument("--updates_per_cycle", type=int, default=1, help="Number of updates each time.")
    # Model arguments
    parser.add_argument("--gpuid", type=int, default=0, help="Which gpu to train on")
    parser.add_argument("--policy", type=str, default="Deterministic", help="Gaussian or deterministic?")
    parser.add_argument("--notes", type=str, default="", help="Additional notes about the training method")
    
    args = parser.parse_args()

    env = FlockEnviroment(args.num_agents)

    # algo = MADDPG(args.num_agents, env.get_observation_size(), env.dimensions, args.batch_size,
    #              args.memory_capacity, args.pre_train_eps)

    
    # algo = SharedMADDPG(args.num_agents, env.get_observation_size(), env.dimensions, args.batch_size,
    #                     args.memory_capacity, args.pre_train_eps)


    kwargs = {"policy": args.policy, "gpuid": args.gpuid} 
    algo = SAC(args.num_agents, env.get_observation_size(), np.array([[0.25, 0.25],[-0.25, -0.25]]), **kwargs)
   
    run_name = "{}_SAC_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), kwargs["policy"], args.notes)
    writer = SummaryWriter('runs/'+run_name)
    
    
    last_10_collisions = np.zeros(10)
    reward_hist = []
    collision_hist = []
    while (algo.episode_done < args.num_episodes):
        print(algo.episode_done, "/", args.num_episodes)
        state = env.reset()
        done = False
        total_reward = 0.0
        
        step_counter = 0
        while(not done):
            actions = algo.select_action(algo.to_float_tensor(state))
            # actions = (0.25-(-0.25))*torch.rand((args.num_agents, 2)) + (-0.25)
            old_state, rewards, state, done = env.step(actions.clone().detach().cpu().numpy())
            total_reward += rewards.sum()
            # algo.memory.push(old_state, actions,\
            #     state, rewards)
            mask = float(not done)
            algo.memory.push(old_state, actions, rewards, state, mask)
            
            if step_counter % args.update_cycle == 0:
                for _ in range(args.updates_per_cycle):
                    algo.update_policy()# new, update every step 
            
            step_counter += 1

        algo.episode_done += 1
        # algo.update_policy()
        # if (algo.episode_done % 10 == 0):
        #     env.display_last_episode()
        print("Collisions", env.episode_collisions)
        print("Total reward", total_reward)
        reward_hist.append(total_reward)
        collision_hist.append(env.episode_collisions)
        writer.add_scalar('reward/train', total_reward, algo.episode_done)
        writer.add_scalar('collision/train', env.episode_collisions, algo.episode_done)
        last_10_collisions[algo.episode_done % 10 - 1] = env.episode_collisions
        if (algo.episode_done % 10 == 0):
            print("--- Last Ten Average", last_10_collisions.mean())
            last_10_collisions = np.zeros(10)

        if (algo.episode_done % 10000 == 0):
            checkpoint_name = run_name + "_" + str(algo.episode_done)
            algo.save_checkpoint(checkpoint_name)

    checkpoint_name = run_name + "_final"
    algo.save_checkpoint(checkpoint_name)
    print("reward history", reward_hist)
    print("first 10 reward", reward_hist[:10])
    print("last 10 reward", reward_hist[-10:])
    print("collision history", collision_hist)
    print("first 10 collision", collision_hist[:10])
    print("last 10 collision", collision_hist[-10:])

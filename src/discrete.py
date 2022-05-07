import sys
import os
import numpy as np
sys.path.append(os.path.relpath("."))
import argparse
from enviroments.flocking_enviroment import *
from sarsa import Sarsa
from qlearning import Qlearning
import datetime
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=40)
    parser.add_argument("--num_episodes", type=int, default=10000000)
    parser.add_argument("--eval_frequency", type=int, default=10, help="")
    parser.add_argument("--eval_episodes", type=int, default=3, help="")
    parser.add_argument("--algorithm", type=str, default=None, help="Which algorithm to run? (SARSA or Q-Learning)")
    parser.add_argument("--comment", type=str, default="", help="Description of this particular run")
    parser.add_argument("--checkpoint_frequency", type=int, default=200, help="Save to checkpoint every how many episodes")
    # Model arguments
    args = parser.parse_args()

    env = FlockEnviroment(args.num_agents)



    run_name = "{}_{}_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.algorithm, str(args.num_agents), args.comment)
    writer = SummaryWriter('runs/'+run_name)



    if args.algorithm == "sarsa":
        algo = Sarsa(args.num_agents, env.get_observation_size(), env.dimensions)
    elif args.algorithm == "qlearning":
        algo = Qlearning(args.num_agents, env.get_observation_size(), env.dimensions)    # last_10_collisions = np.zeros(10)


    all_evals = []
    while (algo.episode_done < args.num_episodes):
        print(algo.episode_done, "/", args.num_episodes)

        ## learning
        state = env.reset()
        done = False
        train_total_reward = 0

        if args.algorithm == "sarsa":
            action = algo.select_action(state)
            while(not done):
                state, reward, next_state, done = env.step(action)
                next_action = algo.select_action(next_state)
                algo.update_policy(state, action, reward, next_state, next_action)
                action = next_action
                train_total_reward += reward.sum()
                # if algo.steps_done % 100 == 0 and algo.steps_done > 0:
                #     algo.update_target()
    
        elif args.algorithm == "qlearning":
            while(not done):
                action = algo.select_action(state)
                state, reward, next_state, done = env.step(action)
                algo.update_policy(state, action, reward, next_state)
                state = next_state
                train_total_reward += reward.sum()
                if algo.steps_done % 100 == 0 and algo.steps_done > 0:
                    algo.update_target()


        # if algo.episode_done % 10 == 0 and algo.episode_done > 0:
        #     algo.update_target()

        writer.add_scalar('reward/train', train_total_reward, algo.episode_done) 
        #if (algo.episode_done % 100 == 0):
            #env.display_last_episode()
        if (algo.episode_done % args.eval_frequency == 0):
            eval_store = []
            for eval in range(args.eval_episodes):
                state = env.reset()
                done = False
                epi_store = []
                while(not done):
                    action = algo.select_action(state, epsilon=0.)
                    state, reward, next_state, done = env.step(action)
                    epi_store.append(np.sum(reward))
                eval_store.append(np.mean(epi_store))
            mean_r = np.mean(eval_store)
            print("Eval R:", mean_r)
            writer.add_scalar('reward/eval', mean_r, algo.episode_done)
            all_evals.append(mean_r)
        algo.episode_done += 1

        if algo.episode_done % args.checkpoint_frequency == 0:
            algo.save_checkpoint(run_name + "_" + str(algo.episode_done))

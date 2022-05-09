import sys
import os
import numpy as np
sys.path.append(os.path.relpath("."))
import argparse
from enviroments.flocking_enviroment import *
# from enviroments._flocking_environment import *
from sarsa import Sarsa
from qlearning import Qlearning
from randpol import RandomPolicy 
import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=40)
    parser.add_argument("--num_episodes", type=int, default=10000000)
    parser.add_argument("--eval_frequency", type=int, default=10, help="")
    parser.add_argument("--eval_episodes", type=int, default=3, help="")
    parser.add_argument("--algorithm", type=str, default=None, help="Which algorithm to run? (SARSA or Q-Learning)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.05, help="epislon greedy parameter")
    parser.add_argument("--hidden", type=int, default=32, help="hidden layer size")
    parser.add_argument("--maneu", type=float, default=0.7, help="maneuverability constant")
    parser.add_argument("--neighbor", type=int, default=8, help="neighbor count")
    parser.add_argument("--comment", type=str, default="", help="Description of this particular run")
    parser.add_argument("--checkpoint_frequency", type=int, default=200, help="Save to checkpoint every how many episodes")
    # Model arguments
    parser.add_argument("--show", dest="show", action="store_true")
    parser.set_defaults(show=False)
    parser.add_argument("--savegif", dest="savegif", action="store_true")
    parser.set_defaults(savegif=False)
    parser.add_argument("--loaded_checkpoint", type=str, default=None, help="checkpoint name")
    args = parser.parse_args()






    env = FlockEnviroment(args.num_agents)
    env.neighbor_view_count = args.neighbor
    env.maneuverability = args.maneu


    if not args.show and args.algorithm != "random":
        run_name = "{}_{}_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.algorithm, str(args.num_agents), args.comment)
        writer = SummaryWriter('realruns/'+run_name)
        dir_path = "../results/" + run_name
        if (os.path.exists(dir_path)):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

        training_data_inds = {"Rewards":0, "Collisions":1, "Density":2, "Density2":3}
        training_data = np.zeros((len(training_data_inds), args.num_episodes))

        eval_data_inds = {"Collisions":0, "Density":1, "Density2":2}
        eval_data = np.zeros((len(eval_data_inds), int(args.num_episodes / args.eval_frequency) + 1))



    if args.algorithm == "sarsa":
        algo = Sarsa(args.num_agents, env.get_observation_size(), env.dimensions, args.gamma, args.hidden)
    elif args.algorithm == "qlearning":
        algo = Qlearning(args.num_agents, env.get_observation_size(), env.dimensions, args.gamma, args.hidden)    # last_10_collisions = np.zeros(10)
    elif args.algorithm == "random":
        algo = RandomPolicy(args.num_agents, env.get_observation_size(), env.dimensions, args.gamma, args.hidden)    # last_10_collisions = np.zeros(10)

        collisions_store = []
        deviation_store = []
        square_deviation_store = []
        for episode in range(args.eval_episodes):
            print(episode, "/", args.eval_episodes)
            state = env.reset()
            done = False
            # epi_store = []
            collisions_store2 = []
            deviation_store2 = []
            square_deviation_store2 = []
            while(not done):
                action = algo.select_action(state)
                state, reward, next_state, done = env.step(action)
                # epi_store.append(np.sum(reward))
            # eval_store.append(np.mean(epi_store))
            collisions_store.append(env.episode_collisions)
            deviation_store.append(env.deviation)
            square_deviation_store.append(env.square_deviation)

        # mean_r = np.mean(eval_store)
        # print("Eval R:", mean_r)
        mean_c = np.mean(collisions_store) / args.num_agents
        mean_d = np.mean(deviation_store) / args.num_agents
        mean_d2 = np.mean(square_deviation_store) / args.num_agents
        print("Eval C:", mean_c)
        print("Eval D:", mean_d)
        print("Eval D2:", mean_d2)
        # writer.add_scalar('reward/eval', mean_r, algo.episode_done)

        exit()

    if args.show:
        algo.load_checkpoint("checkpoints/" + args.loaded_checkpoint)

    

    # all_evals = []
    while (algo.episode_done < args.num_episodes):
        print(algo.episode_done, "/", args.num_episodes)


        if args.show:
            state = env.reset()
            done = False
            # epi_store = []
            total_reward = 0
            while(not done):
                action = algo.select_action(state, epsilon=0.)
                state, reward, next_state, done = env.step(action)
                total_reward += reward.sum()
                # epi_store.append(np.sum(reward))
            # eval_store.append(np.mean(epi_store))
            if args.savegif:
                if not os.path.exists('gif/'):
                    os.makedirs('gif/')
                env.display_last_episode(args.loaded_checkpoint, "gif/" + args.loaded_checkpoint + "_" + str(algo.episode_done) + ".gif")
            else:
                env.display_last_episode(args.loaded_checkpoint)
            algo.episode_done += 1
            continue
            

        ## learning

        state = env.reset()
        done = False
        train_total_reward = 0
        reward_acum = []
        collision_acum = []
        density_acum = []
        exps = []

        if args.algorithm == "sarsa":
            action = algo.select_action(state, epsilon=args.epsilon)
            while(not done):
                state, reward, next_state, done = env.step(action)
                reward_acum.append(np.mean(reward))
                next_action = algo.select_action(next_state, epsilon=args.epsilon)
                algo.update_policy(state, action, reward, next_state, next_action)
                action = next_action
                train_total_reward += reward.sum()
                if algo.steps_done % 100 == 0 and algo.steps_done > 0:
                    algo.update_target()
    
        elif args.algorithm == "qlearning":
            while(not done):
                action = algo.select_action(state, epsilon=args.epsilon)
                state, reward, next_state, done = env.step(action)
                reward_acum.append(np.mean(reward))
                algo.update_policy(state, action, reward, next_state)
                state = next_state
                train_total_reward += reward.sum()
                if algo.steps_done % 100 == 0 and algo.steps_done > 0:
                    algo.update_target()
        
        training_data[training_data_inds["Rewards"], algo.episode_done] = np.sum(reward_acum)
        training_data[training_data_inds["Collisions"], algo.episode_done] = env.episode_collisions
        training_data[training_data_inds["Density"], algo.episode_done] = env.deviation
        training_data[training_data_inds["Density2"], algo.episode_done] = env.square_deviation

        # if algo.episode_done % 10 == 0 and algo.episode_done > 0:
        #     algo.update_target()

        writer.add_scalar('Rewards/train', np.sum(reward_acum), algo.episode_done) 
        writer.add_scalar('Collisions/train', env.episode_collisions, algo.episode_done) 
        writer.add_scalar('Density/train', env.deviation, algo.episode_done) 
        writer.add_scalar('Density2/train', env.square_deviation, algo.episode_done) 
            #if (algo.episode_done % 100 == 0):
                #env.display_last_episode()
        if (algo.episode_done % args.eval_frequency == 0):
            # eval_store = []
            collisions_store = []
            deviation_store = []
            square_deviation_store = []
            for eval in range(args.eval_episodes):
                state = env.reset()
                done = False
                # epi_store = []
                collisions_store2 = []
                deviation_store2 = []
                square_deviation_store2 = []
                while(not done):
                    action = algo.select_action(state, epsilon=0.)
                    state, reward, next_state, done = env.step(action)
                    # epi_store.append(np.sum(reward))
                # eval_store.append(np.mean(epi_store))
                collisions_store.append(env.episode_collisions)
                deviation_store.append(env.deviation)
                square_deviation_store.append(env.square_deviation)

            # mean_r = np.mean(eval_store)
            # print("Eval R:", mean_r)
            mean_c = np.mean(collisions_store) / args.num_agents
            mean_d = np.mean(deviation_store) / args.num_agents
            mean_d2 = np.mean(square_deviation_store) / args.num_agents
            eval_data[eval_data_inds["Collisions"], int(algo.episode_done / args.eval_frequency)] = mean_c
            eval_data[eval_data_inds["Density"], int(algo.episode_done / args.eval_frequency)] = mean_d
            eval_data[eval_data_inds["Density2"], int(algo.episode_done / args.eval_frequency)] = mean_d2
            print("Eval C:", mean_c)
            print("Eval D:", mean_d)
            print("Eval D2:", mean_d2)
            # writer.add_scalar('reward/eval', mean_r, algo.episode_done)
            writer.add_scalar('Collisions/eval', mean_c, algo.episode_done)
            writer.add_scalar('Density/eval', mean_d, algo.episode_done)
            writer.add_scalar('Density2/eval', mean_d2, algo.episode_done)
            # all_evals.append(mean_r)

        if algo.episode_done % args.checkpoint_frequency == 0:
            algo.save_checkpoint(run_name + "_" + str(algo.episode_done))
            # env.display_last_episode(run_name + ": " + str(algo.episode_done), dir_path + "/fig" + str(algo.episode_done) + ".gif")
        
        algo.episode_done += 1
    
    if not args.show:
        algo.save_checkpoint(run_name + "_final")
        """
        for key in training_data_inds:
            fig, axes = plt.subplots(1, 1)
            #fig.tight_layout()
            axes.set_title("Training: " + key)
            axes.set_xlabel("Episode")
            axes.plot(training_data[training_data_inds[key]])
            plt.savefig(dir_path + "/" + key + ".png")
            plt.close(fig)
        """
        with open(dir_path + "/training_keys.txt", "w") as fref:
            keys = [None] * len(training_data_inds)
            for key in training_data_inds:
                keys[training_data_inds[key]] = key
            fref.write("\n".join(keys))
        np.save(dir_path + "/" + "training_data.npy", training_data)
        """
        for key in eval_data_inds:
            fig, axes = plt.subplots(1, 1)
            #fig.tight_layout()
            axes.set_title("Evaluation: " + key)
            axes.set_xlabel("Episode")
            axes.plot(range(0, args.num_episodes + 1, args.eval_frequency), eval_data[eval_data_inds[key]])
            plt.savefig(dir_path + "/" + key + ".png")
            plt.close(fig)
        """
        with open(dir_path + "/eval_keys.txt", "w") as fref:
            keys = [None] * len(eval_data_inds)
            for key in eval_data_inds:
                keys[eval_data_inds[key]] = key
            fref.write("\n".join(keys))
        np.save(dir_path + "/" + "eval_data.npy", eval_data)

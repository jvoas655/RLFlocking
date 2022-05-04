import numpy as np
import argparse
from enviroments.flocking_enviroment import *
from models.discrete.value import QApproximationWithNN
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils.helpers import get_device


def Sarsa(
    env, # openai gym environment
    gamma:float, # discount factor
    num_agents:int, # number of birds
    alpha:float, # step size
    QNet:QApproximationWithNN,
    num_episode:int,
    eps:float,
    writer, # tensorboard writer
    checkpoint_name:str,
    test=False
) -> np.array:
    """
    Implement Sarsa
    """

    norm = 0.1
    action_map = {0: np.array([0., 0.]),
                  1: np.array([0., 1.]) * norm,
                  2: np.sqrt(0.5) * (np.array([1., 1.])) * norm,
                  3: np.array([1., 0.]) * norm,
                  4: np.sqrt(0.5) * (np.array([1., -1.])) * norm,
                  5: np.array([0., -1.]) * norm,
                  6: np.sqrt(0.5) * (np.array([-1., -1.])) * norm,
                  7: np.array([-1., 0.]) * norm,
                  8: np.sqrt(0.5) * (np.array([-1., 1.])) * norm}

    def epsilon_greedy_policy(num_agents, state, epsilon=.0):
        action = np.zeros((num_agents, env.dimensions))
        for agent in range(num_agents):
            s = state[agent]
            Q = [QNet(s, action_map[a]) for a in action_map]
            if np.random.rand() < epsilon:
                act_idx = np.random.randint(len(action_map))
                action[agent] = action_map[act_idx]
            else:
                action[agent] = action_map[np.argmax(Q)]

        return action


    #TODO: implement this function
    # raise NotImplementedError()
    # collision_hist = []
    
    reward_hist = []
    for episode in range(num_episode):
        print(episode, "/", num_episode)
        state = env.reset()
        action = epsilon_greedy_policy(num_agents, state)
        total_reward = 0.0
        while True:
            # env.render()
            state, rewards, state_, done = env.step(action) # S, A, R, S'
            total_reward += rewards.sum()
            if test:
                action_ = epsilon_greedy_policy(num_agents, state_, epsilon=0.) # S, A, R, S', A'
            else:
                action_ = epsilon_greedy_policy(num_agents, state_, epsilon=eps) # S, A, R, S', A'
           
                for agent in range(num_agents):
                    target = rewards[agent] + gamma * QNet(state_[agent], action_[agent]) - QNet(state[agent], action[agent])
                    QNet.update(state[agent], action[agent], target) 

            action = action_

            if done:
                break
        print("collisions:", env.episode_collisions)
        print("reward:", total_reward)
        reward_hist.append(total_reward)

        if not test:
            writer.add_scalar('collision/train', env.episode_collisions, episode)
            writer.add_scalar('reward/train', total_reward, episode)
            # collision_hist.append(env.episode_collisions)
            if (episode % 100 == 0):
                QNet.save_checkpoint("SARSA", checkpoint_name + "_" + str(episode))

        # if (episode % 100 == 0):
        #     env.display_last_episode()
        # if test:
        #     env.display_last_episode()
    if not test:
        QNet.save_checkpoint("SARSA", checkpoint_name + "_final")
    else:
        print("average reward", sum(reward_hist) / len(reward_hist))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=40)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--gpuid", type=int, default=None)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--notes", type=str, default="")

    parser.add_argument("--test", dest="test", action="store_true")
    parser.set_defaults(test=False)
    parser.add_argument("--loaded_checkpoint", type=str, default=None, help="path to checkpoint")
    args = parser.parse_args()

    if not args.test:
        run_name = "{}_SARSA_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), str(args.num_agents), args.notes)
        writer = SummaryWriter('runs/'+run_name)
        checkpoint_name = run_name
    else:
        writer = None
        checkpoint_name = None

    env = FlockEnviroment(args.num_agents)
    QNet = QApproximationWithNN(num_agents=args.num_agents, state_dims=env.get_observation_size(), action_dims=env.dimensions, device=get_device(args.gpuid))
    if args.test:
        QNet.load_checkpoint(args.loaded_checkpoint)
    Sarsa(env, gamma=0.9, num_agents=args.num_agents, alpha=0.01, QNet=QNet, num_episode=args.num_episodes, eps=args.eps, writer=writer, checkpoint_name = checkpoint_name, test=args.test)

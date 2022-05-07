import numpy as np
import argparse
from enviroments.flocking_enviroment import *
from enviroments.flocking_environment2 import *
from models.discrete.value import QApproximationWithNN
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils.helpers import get_device
from memory import ReplayMemory, Experience
from models.discrete.prioritized_memory import PrioritizedMemory
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--num_agents", type=int, default=40)
parser.add_argument("--num_episodes", type=int, default=1000)
parser.add_argument("--gpuid", type=int, default=None)
parser.add_argument("--notes", type=str, default="")
parser.add_argument("--replay", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=256)
 
parser.add_argument("--test", dest="test", action="store_true")
parser.set_defaults(test=False)
parser.add_argument("--loaded_checkpoint", type=str, default=None, help="path to checkpoint")

args = parser.parse_args()

def QLearning(
    env, # openai gym environment
    gamma:float, # discount factor
    num_agents:int, # number of birds
    alpha:float, # step size
    QNet:QApproximationWithNN,
    num_episode:int,
    writer, # tensorboard writer
    checkpoint_name # checkpoint name
) -> np.array:

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
    
    def soft_update(target, source, t):
        for target_param, source_param in zip(target.model.parameters(),
                                              source.model.parameters()):
            target_param.data.copy_(
                (1 - t) * target_param.data + t * source_param.data)


    def hard_update(target, source):
        for target_param, source_param in zip(target.model.parameters(),
                                              source.model.parameters()):
            target_param.data.copy_(source_param.data)

    def maxQ(s, QNet_target):
        maxq = -np.inf
        for a in action_map:
            qval = QNet_target(s, action_map[a])
            if qval > maxq:
                maxq = qval
        return maxq

    def batchMaxQ(S, QNet_target):
        maxQ = np.ones(S.shape[0]) * (-np.inf)
        for i in range(S.shape[0]):
            for a in action_map:
                qval = QNet_target(S[i], action_map[a])
                if qval > maxQ[i]:
                    maxQ[i] = qval
        return maxQ


    #TODO: implement this function
    # raise NotImplementedError()
    # collision_hist = []
    

    if args.replay == "random":
        replay_buffer = ReplayMemory(10000)
    elif args.replay == "prioritized":
        replay_buffer = PrioritizedMemory(10000)

    QNet_target = deepcopy(QNet)
    
    
    reward_hist = []

    for episode in range(num_episode):
        print(episode, "/", num_episode)
        state = env.reset()
        total_reward = 0.0
        while True:
            # env.render()
            if args.test:
                eps = 0.0
            else:
                eps = 0.1
            action = epsilon_greedy_policy(num_agents, state, epsilon=eps) 
            state, rewards, state_, done = env.step(action)
            
            total_reward += rewards.sum()
            if not args.test:
                for agent in range(num_agents):
                    target = rewards[agent] + gamma * maxQ(state_[agent], QNet_target) #  - QNet(state[agent], action[agent]) # bug...
                    QNet.update(state[agent], action[agent], np.array(target))
                    if args.replay == "prioritized":
                        err = abs(target - QNet(state[agent], action[agent]))
                        replay_buffer.add(err, (state[agent], action[agent], rewards[agent], state_[agent], done))
                    elif args.replay == "random":
                        replay_buffer.push(state[agent], action[agent], state_[agent], rewards[agent])
                if args.replay == "prioritized" and episode > 5:
                    mini_batch, idxs, is_weights = replay_buffer.sample(args.batch_size)
                    mini_batch = np.array(mini_batch).transpose()
                    batch_state = np.vstack(mini_batch[0])
                    batch_action = np.vstack(mini_batch[1])# list(mini_batch[1])
                    batch_reward = np.array(list(mini_batch[2]))
                    batch_next_state = np.vstack(mini_batch[3])
                    dones = mini_batch[4]
                    dones = np.array(dones.astype(int))
                    
                    batch_target = batch_reward + (1 - dones) * gamma * batchMaxQ(batch_next_state, QNet_target)
                    QNet.update(batch_state, batch_action, np.expand_dims(batch_target, 1))
                    batch_error = np.abs(QNet(batch_state, batch_action).squeeze(1) - batch_target)
                    for i in range(args.batch_size):
                        idx = idxs[i]
                        replay_buffer.update(idx, batch_error[i])



                elif args.replay == "random" and episode > 5:
                    transitions = replay_buffer.sample(args.batch_size)
                    batch = Experience(*zip(*transitions))

                    batch_state = np.array(list(batch.states))
                    batch_action = np.array(list(batch.actions))
                    batch_reward = np.array(list(batch.rewards))
                    batch_next_state = np.array(list(batch.next_states))

                    batch_target = batch_reward + gamma * batchMaxQ(batch_next_state, QNet_target)
                    QNet.update(batch_state, batch_action, batch_target)
                     
            

            state = state_
            if done:
                break
        
        print("collisions:", env.episode_collisions)
        print("reward:", total_reward)
        if not args.test:
            writer.add_scalar('collision/train', env.episode_collisions, episode)
            writer.add_scalar('reward/train', total_reward, episode)
            
            if (episode % 500 == 0):
                QNet.save_checkpoint("Qlearning", checkpoint_name + "_" + str(episode))

            if (episode % 10 == 0 and episode != 0):
                soft_update(QNet_target, QNet, 0.01)

        # if (episode % 100 == 0):
        #     env.display_last_episode()
        reward_hist.append(total_reward)
        if args.test:
            env.display_last_episode()

    if not args.test: 
        QNet.save_checkpoint("Qlearning", checkpoint_name + "_final")
    else: 
        print("average reward:", sum(reward_hist) / len(reward_hist))

if __name__ == "__main__":
    env = FlockEnviroment(args.num_agents)
   

    QNet = QApproximationWithNN(num_agents=args.num_agents, state_dims=env.get_observation_size(), action_dims=env.dimensions, device=get_device(args.gpuid))
    
    if not args.test:
        run_name = "{}_QLEARNING_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), str(args.num_agents), args.notes)
        writer = SummaryWriter('runs/'+run_name)
        checkpoint_name = run_name

        QLearning(env, gamma=0.9, num_agents=args.num_agents, alpha=0.01, QNet=QNet, num_episode=args.num_episodes, writer=writer, checkpoint_name=checkpoint_name)

    else:
        QNet.load_checkpoint(args.loaded_checkpoint)
        QLearning(env, gamma=0.9, num_agents=args.num_agents, alpha=0.01, QNet=QNet, num_episode=args.num_episodes, writer=None, checkpoint_name=None)


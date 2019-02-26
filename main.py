from bit_flip_env import BitFlipEnv
from utils import *
from dqn import DQN
import time
import torch
import argparse
from itertools import count
import random
from evaluate_model import eval_model
import numpy as np


def select_action(args, state, goal, actions_num, policy_net, steps_done, device):
    sample = random.random()
    eps_threshold = max(args.eps_end,
                        args.eps_start * (1 - steps_done / args.eps_decay) +
                        args.eps_end * (steps_done / args.eps_decay))
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(torch.cat((state, goal), 1)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(actions_num)]], device=device, dtype=torch.long)


def optimize_model(args, policy_net, target_net, optimizer, memory, device):
    # First check if there is an available batch
    if len(memory) < args.batch_size:
        return 0

    # Sample batch to learn net(from
    transitions = memory.sample(args.batch_size)

    # Transpose the batch
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state, dim=0)
    action_batch = torch.cat(batch.action, dim=0)
    reward_batch = torch.cat(batch.reward, dim=0)

    # Compute Q(s_t, a) - the model computes Q(s_t),
    # Then, using gather, we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros((args.batch_size, 1), device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].unsqueeze(1).detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch.float()

    # Compute Huber loss
    loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
    if args.reg_param > 0:
        regularization = 0.0
        for param in policy_net.parameters():
            regularization += torch.sum(torch.abs(param))
        loss += args.reg_param * regularization

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reset Bit-Flip environment
    env = BitFlipEnv(size=args.state_size, shaped_reward=args.shaped_reward, dynamic=args.dynamic)
    actions_num = env.size
    states_dim = env.size

    # Statistics
    steps_done = 0
    reset_target_cnt = 0
    episode_durations = []
    eval_reward_array = []
    success_ratio_array = []

    # Experience Replay: memory of available transitions
    memory = ReplayMemory(100000)

    # Create DQN models
    policy_net = DQN(states_dim * 2, args.hidden_dim, actions_num, dropout_rate=args.dropout).to(device)
    target_net = DQN(states_dim * 2, args.hidden_dim, actions_num, dropout_rate=args.dropout).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.alpha)

    # Failed episodes buffer for DHER
    failed_episodes = []
    failed_episodes_size = 10
    failed_episodes_index = 0

    # Main loop
    for i_episode in range(args.episodes):
        # Initialize the environment and state
        state = torch.tensor(env.reset(), device=device, dtype=torch.float).unsqueeze(0)
        goal = torch.tensor(env.target, device=device, dtype=torch.float).unsqueeze(0)

        episode_reward = 0
        episode_memory = []
        for t in count():
            # Select and perform an action
            action = select_action(args, state, goal, actions_num, policy_net, steps_done, device)
            next_state, reward, done = env.step(action.item())

            old_goal = goal.clone()

            # In the dynamic goal case, the goal might change every step
            goal = torch.tensor(env.target, device=device, dtype=torch.float).unsqueeze(0)

            # Next state to tensor
            next_state = torch.tensor(next_state, device=device, dtype=torch.float).unsqueeze(0)

            steps_done += 1
            reset_target_cnt += 1
            episode_reward += reward

            reward = torch.tensor([reward], device=device).unsqueeze(0)

            # Store the transition in memory
            if not (done and reward < 0):
                episode_memory.append((state, action, next_state, reward, old_goal, goal))

            # Move to the next state
            state = next_state

            # Update the target network
            if reset_target_cnt > args.target_update:
                reset_target_cnt = 0
                target_net.load_state_dict(policy_net.state_dict())

            # Episode duration statistics
            if done:
                episode_durations.append(t + 1)
                if reward < 0:
                    # This is a failed trajectory
                    if len(failed_episodes) < failed_episodes_size:
                        failed_episodes.append(episode_memory)
                    else:
                        failed_episodes[failed_episodes_index] = episode_memory
                        failed_episodes_index = (failed_episodes_index + 1) % failed_episodes_size
                break

        # Experience Replay
        for t in range(len(episode_memory)):
            state, action, next_state, reward, old_goal, goal = episode_memory[t]

            state_memory = torch.cat((state, goal), 1)
            if torch.all(next_state == goal):
                next_state_memory = None
            else:
                next_state_memory = torch.cat((next_state, goal), 1)

            memory.push(state_memory, action, next_state_memory, reward)

            #HER
            if args.HER:
                for g in range(args.goals):
                    future_goal = np.random.randint(t, len(episode_memory))
                    _, _, new_goal, _, _, _ = episode_memory[future_goal]
                    state_memory = torch.cat((state, new_goal), 1)
                    if torch.all(next_state == new_goal):  # Done
                        next_state_memory = None
                        reward = torch.zeros(1, 1)
                    else:
                        next_state_memory = torch.cat((next_state, new_goal), 1)
                        reward = torch.zeros(1, 1) - 1.0
                    memory.push(state_memory, action, next_state_memory, reward)

        # DHER
        if args.DHER:
            finish = False
            for i_ep, failed_ep_i in enumerate(failed_episodes):
                for j_ep, failed_ep_j in enumerate(failed_episodes):
                    if i_ep == j_ep:
                        continue
                    for i_i, t_i in enumerate(failed_ep_i):
                        for j_j, t_j in enumerate(failed_ep_j):
                            # Checks if the ith episode's next state is the same as the jth episode's next goal
                            if torch.all(t_i[2] == t_j[5]):
                                m = min(i_i, j_j)
                                for t in range(m, -1, -1):
                                    new_current_goal = failed_ep_j[j_j - t][4]
                                    new_next_goal = failed_ep_j[j_j - t][5]
                                    next_state = failed_ep_i[i_i - t][2]
                                    if torch.all(next_state == new_next_goal):
                                        next_state_memory = None
                                        reward = torch.zeros(1, 1)
                                    else:
                                        next_state_memory = torch.cat((next_state, new_next_goal), 1)
                                        reward = torch.zeros(1, 1) - 1.0
                                    state_memory = torch.cat((failed_ep_i[i_i - t][0], new_current_goal), 1)
                                    action = failed_ep_i[i_i - t][1]
                                    memory.push(state_memory, action, next_state_memory, reward)
                                finish = True
                            if finish:
                                break
                        if finish:
                            break
                    if finish:
                        break

        # Perform one step of the optimization (on the target network)
        optimization_steps = 5
        loss = 0.0
        for _ in range(optimization_steps):
            loss += optimize_model(args, policy_net, target_net, optimizer, memory, device)
        loss /= optimization_steps

        # Episodes statistics
        if i_episode % 10 == 0 and i_episode != 0:
            print("Evaluation:")
            eval_reward, success_ratio = eval_model(model=policy_net, env=env, episodes=10, device=device)
            eval_mean_reward = np.mean(eval_reward)
            eval_reward_array.append(eval_mean_reward)
            success_ratio_array.append(success_ratio)

        print("Episode %d complete, episode duration = %d, loss = %.3f, reward = %d" %
              (i_episode, episode_durations[-1], loss, episode_reward))

        torch.save(policy_net.state_dict(), 'bit_flip_model.pkl')

    np.save("success_ratio.npy", success_ratio_array)
    print('Complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)

    # Run parameters
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size to train on')
    parser.add_argument('--episodes', type=int, default=5000, help='Amount of train episodes to run')
    parser.add_argument('--state-size', type=int, default=4, help='Size of the environments states')
    parser.add_argument('--shaped-reward', action="store_true")

    # Model arguments
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--hidden-dim', type=int, default=500, help='Dimension of the hidden layer')

    # Optimizer arguments
    parser.add_argument('--alpha', type=float, default=0.001, help='Alpha - Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma - discount factor')
    parser.add_argument('--target-update', type=int, default=500, help='Number of steps until updating target network')
    parser.add_argument('--reg-param', type=float, default=0, help='L1 regulatization parameter')

    # Epsilon - Greedy arguments
    parser.add_argument('--eps-start', type=float, default=1.0, help='Starting epsilon - in epsilon greedy method')
    parser.add_argument('--eps-end', type=float, default=0.1,
                        help='Final epsilon - in epsilon greedy method. When epsilon reaches this value it will stay')
    parser.add_argument('--eps-decay', type=int, default=500000,
                        help='Epsilon decay - how many steps until decaying to the final epsilon')

    # Hindsight Experience Replay (HER) arguments
    parser.add_argument('--HER', action="store_true", help="Use the HER algorithm")
    parser.add_argument('--goals', type=int, default=4, help="Number of goals for the HER algorithm")

    # Dynamic Hindsight Experience Replay (DHER) arguments
    parser.add_argument('--DHER', action="store_true", help="Use the HER algorithm")
    parser.add_argument('--dynamic', action="store_true", help="Use the dynamic mode for the bit flip environment")

    args = parser.parse_args()

    start_time = time.time()
    train(args)
    print('Run finished successfully in %s seconds' % round(time.time() - start_time))

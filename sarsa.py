'''
Implements Q Learning to find an optimal policy for a provided MDP
'''

import numpy as np
import argparse
import os
from tqdm import tqdm

from mdp import GridWorldCarry


def initialize_Q(state_size, action_size):
    return np.zeros((state_size, action_size))

def choose_action(Q, state, epsilon, action_space):
    # with epsilon probability choose a random action
    if np.random.rand() < epsilon:
        action = np.random.randint(0, action_space)
    # otherwise choose from the best available actions
    else:
        best_action_value = -np.inf
        best_actions = []
        for a in range(action_space):
            value = Q[state,a]
            if value > best_action_value:
                best_action_value = value
                best_actions = [a]
            elif value == best_action_value:
                best_actions.append(a)
        action = best_actions[np.random.randint(0, len(best_actions))]
    return action

def sarsa(Q, task, n_episodes, alpha, gamma, epsilon):
    start_state = task.state_number
    
    for episode in range(n_episodes):
        reward = 0.

        state = task.state_number
        action = choose_action(Q, state, epsilon, task.action_space)

        total_reward = 0.
        steps = 0

        # Terminate an episode if there is positive reward (i.e. we reached a terminal state)
        while not reward > 0.:
            steps += 1
            new_state = task.transition(state, action)
            reward = task.reward(new_state)
            total_reward += reward
            new_action = choose_action(Q, new_state, epsilon, task.action_space)

            # Compute TD error
            delta = reward + gamma*Q[new_state, new_action] - Q[state, action]
            # Update Q
            Q[state, action] = Q[state, action] + alpha*delta

            state = new_state
            action = new_action
        
        # reset state at end of each episode
        task.set_state(start_state)

        print(f"Episode {episode+1}: {steps} steps, {total_reward} reward")

    return Q

def optimal_policy_actions(Q, task):
    all_actions = []

    state = task.state_number

    steps = 0
    reward = 0.
    while not reward > 0.:
        steps += 1
        action = choose_action(Q, state, 0., task.action_space)
        all_actions.append(action)
        new_state = task.transition(state, action)
        reward = task.reward(new_state)
        state = new_state
    
    return all_actions

def show_actions(task, action_sequence):
    start_state = task.state_number

    state = task.state_number
    task.render_state(start_state)
    task.show(ms=2000)

    for action in tqdm(action_sequence):
        state = task.transition(state, action)
        task.set_state(state)
        task.render_state(state)
        task.show(ms=500)
    
    task.set_state(start_state)


def view_policy(Q, task):
    action_sequence = optimal_policy_actions(Q, task)
    num_steps = show_actions(task, action_sequence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q-Learning on an MDP")
    parser.add_argument("--alpha", type=float, default=0.1, help="The learning rate for the SARSA algorithm")
    parser.add_argument("--gamma", type=float, default=0.8, help="The discount factor for the SARSA algorithm")
    parser.add_argument("--epsilon", type=float, default=0.1, help="The exploration factor for the SARSA algorithm")
    parser.add_argument("--n_episodes", type=int, default=100, help="The number of episodes to train for using the SARSA algorithm")
    parser.add_argument("--save-dir", type=str, default="F:\\Brown\\cs2951x\\trained_models", help="The directory where model weights can be saved to and loaded from")
    parser.add_argument("-l", "--load", action="store_true", help="Load Q from")
    parser.add_argument("-T", "--train", action="store_true", help="Train Q on the task")
    parser.add_argument("-v", "--view", action="store_true", help="View the optimal policy stored in Q")
    parser.add_argument('-n', "--name", required=True, help="Name of this experiment")
    args = parser.parse_args()

    task = GridWorldCarry([(0,0), (1,2), (5,6), (4,2)], (2,2), (0,3), (3,0), board_size=7)
    model_file = os.path.join(args.save_dir, f"{args.name}.npy")

    if args.load:
        Q = np.load(model_file)
    else:
        Q = initialize_Q(task.state_space, task.action_space)

    if args.train:
        Q = sarsa(Q, task, args.n_episodes, args.alpha, args.gamma, args.epsilon)
        np.save(model_file, Q)

    if args.view:
        view_policy(Q, task)

    



            
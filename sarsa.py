'''
Implements Q Learning to find an optimal policy for a provided MDP
'''

import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

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

    rewards = []
    
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
        rewards.append(total_reward)
        print(f"Episode {episode+1}: {steps} steps, {total_reward} reward")

    return Q, rewards

def log_results(rewards, exp_name, n_episodes):
    '''
    Plot the reward per episode and write to file
    '''
    f = plt.figure()
    ax = f.add_subplot(111)
    
    reward_arr = np.zeros((len(rewards), n_episodes))

    for i, series in enumerate(rewards):
        reward_arr[i] = np.array(series)
    
    third_quartile = np.quantile(reward_arr, 0.75, axis=0)
    first_quartile = np.quantile(reward_arr, 0.25, axis=0)
    median = np.median(reward_arr, axis=0)
    
    xs = np.arange(n_episodes)

    plt.yscale("symlog")
    ax.fill_between(xs, first_quartile, third_quartile)
    ax.plot(xs, median)

    plt.show()


def optimal_policy_actions(Q, task):
    all_actions = []
    all_states = []

    state = task.state_number
    all_states.append(state)

    steps = 0
    reward = 0.
    while not reward > 0.:
        steps += 1
        action = choose_action(Q, state, 0., task.action_space)
        all_actions.append(action)
        new_state = task.transition(state, action)
        reward = task.reward(new_state)
        state = new_state
        all_states.append(state)
    
    return all_actions, all_states

def get_macros_from_action_sequences(sequences, n_macros=5, max_macro_size=10):

    macros_ordered = []
    macro_counts = {}

    # loop through each potential macro
    for l in range(2, max_macro_size+1):
        for seq in sequences:
            for start in range(len(seq)-l):
                macro = tuple(seq[start:start+l])
                if not macro in macro_counts:
                    macro_counts[macro] = 1
                    macros_ordered.append(macro)
                else:
                    macro_counts[macro] += 1

    most_common_macro = None
    highest_count = 0
    for macro in macros_ordered:
        if most_common_macro is None:
            most_common_macro = macro
            highest_count = macro_counts[macro]
        else:
            count = macro_counts[macro]
            if count >= highest_count:
                most_common_macro = macro
                highest_count = count

    print(f"Best macro has length {len(most_common_macro)} and appears {highest_count} times")

    return most_common_macro
    

def show_states(task, state_sequence, name):
    dir = os.path.join("..", "outputs")
    if not os.path.exists(dir):
        os.mkdir(dir)
    writer = imageio.get_writer(os.path.join(dir,f'{name}.mp4'), fps=20)
    for s in tqdm(state_sequence):
        task.render_state(s)
        task.show(ms=(2000 if s==0 or s==len(state_sequence) else 500))
        im = imageio.imread(task.visual_file)
        frames = 40 if s==0 or s==len(state_sequence) else 10
        for _ in range(frames):
            writer.append_data(im)
    writer.close()

def show_actions(task, action_sequence, name):
    dir = os.path.join("..", "outputs")
    if not os.path.exists(dir):
        os.mkdir(dir)
    writer = imageio.get_writer(os.path.join(dir,f'{name}_macro.mp4'), fps=20)
    start_state = task.state_number

    state = task.state_number
    task.render_state(start_state)
    task.show(ms=2000)
    im = imageio.imread(task.visual_file)
    for _ in range(40):
        writer.append_data(im)

    for action in tqdm(action_sequence):
        state = task.transition(state, action)
        task.set_state(state)
        task.render_state(state)
        task.show(ms=500)
        im = imageio.imread(task.visual_file)
        for _ in range(10):
            writer.append_data(im)

    task.set_state(start_state)


def view_policy(Q, task, name):
    _, state_sequence = optimal_policy_actions(Q, task)
    show_states(task, state_sequence, name)

def view_macro(macro, name):
    macro_visualization_task = GridWorldCarry([], (0,0), (5,5), (0,1), board_size=11, action_success_prob=1.0)
    show_actions(macro_visualization_task, macro, name)


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
    parser.add_argument("-m", "--macro", action="store_true", help="Show best macro")
    parser.add_argument('-n', "--name", required=True, help="Name of this experiment")
    args = parser.parse_args()

    task = GridWorldCarry([(0,0), (1,2), (4,2)], (2,2), (0,3), (3,0), board_size=5)
    model_file = os.path.join(args.save_dir, f"{args.name}.npy")

    if args.load:
        Q = np.load(model_file)
    else:
        Q = initialize_Q(task.state_space, task.action_space)

    if args.train:
        all_rewards = []
        for i in range(16):
            print(f"++++++++++++++++++ ROUND {i} ++++++++++++++++++++")
            Q = initialize_Q(task.state_space, task.action_space)
            Q, rewards = sarsa(Q, task, args.n_episodes, args.alpha, args.gamma, args.epsilon)
            np.save(model_file, Q)
            all_rewards.append(rewards)
        log_results(all_rewards, args.name, args.n_episodes)

    if args.view:
        view_policy(Q, task, args.name)

    if args.macro:
        actions, _ = optimal_policy_actions(Q, task)
        print(actions)
        best_macro = get_macros_from_action_sequences([actions])
        print(best_macro)
        view_macro(best_macro, args.name)

    



            
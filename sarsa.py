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
        goal_met = False
        while not goal_met:
            steps += 1
            new_state, new_reward, goal_met = task.transition(state, action)
            total_reward += new_reward
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

def sarsa_lambda(Q, task, n_episodes, alpha, gamma, epsilon, lmbda,e_trace_cutoff = 0.001, verbose=True):
    start_state = task.state_number
    
    rewards = []
    
    it = tqdm(range(n_episodes)) if not verbose else range(n_episodes)

    for episode in it:
        eligibility_trace = {}
        reward = 0.

        state = task.state_number
        action = choose_action(Q, state, epsilon, task.action_space)

        total_reward = 0.
        steps = 0

        # Cap on how long one episode can be
        cutoff = 100000

        goal_met = False
        reward = 0.
        # Terminate an episode if there is positive reward (i.e. we reached a terminal state)
        while not goal_met and steps < cutoff:
            steps += 1
            new_state, reward, goal_met = task.transition(state, action)
            # total_reward += new_reward
            # reward = task.reward(new_state)
            total_reward += reward
            new_action = choose_action(Q, new_state, epsilon, task.action_space)

            # Compute TD error
            delta = reward + gamma*Q[new_state, new_action] - Q[state, action]

            # update eligibility trace of current state, action pair
            if (state, action) in eligibility_trace:
                eligibility_trace[(state,action)] += 1.
            else:
                eligibility_trace[(state,action)] = 1.

            # update Q and eligibility trace for each S,A pair
            # for s in range(Q.shape[0]):
            #     for a in range(Q.shape[1]):
            keys_to_remove = []

            for s,a in eligibility_trace:                
                Q[s, a] = Q[s,a] + alpha*delta*eligibility_trace[(s,a)]
                eligibility_trace[(s,a)] = gamma * lmbda * eligibility_trace[(s,a)]
                if eligibility_trace[(s,a)] < e_trace_cutoff:
                    keys_to_remove.append((s,a))
            
            for key in keys_to_remove:
                eligibility_trace.pop(key)

            state = new_state
            action = new_action
        # print(f"steps: {steps}   :   {goal_met}")
        
        # reset state at end of each episode
        task.set_state(start_state)
        rewards.append(total_reward)
        if verbose:
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
        reward_arr[i] = np.array(series) - 100.
    
    third_quartile = np.quantile(reward_arr, 0.75, axis=0)
    first_quartile = np.quantile(reward_arr, 0.25, axis=0)
    median = np.median(reward_arr, axis=0)
    
    xs = np.arange(n_episodes)

    plt.yscale("symlog")
    # ax.fill_between(xs, first_quartile, third_quartile)
    ax.plot(xs, median)

    plt.show()


def optimal_policy_actions(Q, task):
    all_actions = []
    all_states = []

    state = task.state_number
    all_states.append(state)

    steps = 0
    goal_met = False
    while not goal_met:
        steps += 1
        if steps > 100:
            return [],[]
        action = choose_action(Q, state, 0., task.action_space)
        all_actions.append(action)
        new_state, _, goal_met = task.transition(state, action)
        # reward = task.reward(new_state)
        state = new_state
        all_states.append(state)
    
    return all_actions, all_states

def get_macros_from_action_sequences(sequences, n_macros=5, max_macro_size=10, scale_by_len=False):
    macro_counts = {}

    # loop through each potential macro
    for l in range(2, max_macro_size+1):
        for seq in sequences:
            for start in range(len(seq)-l):
                macro = tuple(seq[start:start+l])
                if not macro in macro_counts:
                    macro_counts[macro] = 1
                else:
                    macro_counts[macro] += 1

    # most_common_macro = None
    # highest_count = 0
    # for macro in macros_ordered:
    #     if most_common_macro is None:
    #         most_common_macro = macro
    #         highest_count = macro_counts[macro]
    #     else:
    #         count = macro_counts[macro]
    #         if count >= highest_count:
    #             most_common_macro = macro
    #             highest_count = count

    macros = []
    counts = []
    scaled_counts = []
    for macro in macro_counts:
        macros.append(macro)
        counts.append(macro_counts[macro])
        scaled_counts.append(macro_counts[macro]*len(macro))

    
    best_macros = []
    best_counts = []

    scores = np.array(scaled_counts) if scale_by_len else np.array(counts)

    for _ in range(n_macros):
        next_best = np.argmax(scores)
        best_macros.append(macros[next_best])
        best_counts.append(counts[next_best])
        scores[next_best] = -1.


    print(f"Best macro has length {len(best_macros[0])} and appears {best_counts[0]} times")
    print(f"The best macro was {best_macros[0]}")

    return best_macros

def get_macros_from_action_sequences_recur(sequences, n_macros=5, max_macro_size=10, scale_by_len=False, action_count=10):
    macro_counts = {}

    # loop through each potential macro
    for l in range(2, max_macro_size+1):
        for seq in sequences:
            for start in range(len(seq)-l):
                macro = tuple(seq[start:start+l])
                if not macro in macro_counts:
                    macro_counts[macro] = 1
                else:
                    macro_counts[macro] += 1

    # most_common_macro = None
    # highest_count = 0
    # for macro in macros_ordered:
    #     if most_common_macro is None:
    #         most_common_macro = macro
    #         highest_count = macro_counts[macro]
    #     else:
    #         count = macro_counts[macro]
    #         if count >= highest_count:
    #             most_common_macro = macro
    #             highest_count = count
    
    # if the action sequences are length 1 or less, we can't extract any more macros
    if len(macro_counts) == 0:
        return []

    macros = []
    counts = []
    scaled_counts = []
    for macro in macro_counts:
        macros.append(macro)
        counts.append(macro_counts[macro])
        scaled_counts.append(macro_counts[macro]*len(macro))

    
    best_macros = []
    best_counts = []

    scores = np.array(scaled_counts) if scale_by_len else np.array(counts)
    # print(scores)
    next_best = np.argmax(scores)
    best_macros.append(macros[next_best])
    best_counts.append(counts[next_best])

    # replace subsequences with new macro
    new_macro = list(macros[next_best])
    new_sequences = []
    for seq in sequences:
        new_seq = []
        i = 0
        while i < len(seq):
            if len(seq) - i < len(new_macro):
                new_seq.append(seq[i])
                i+=1
            else:
                if seq[i:i+len(new_macro)] == new_macro:
                    new_seq.append(action_count)
                    i += len(new_macro)
                else:
                    new_seq.append(seq[i])
                    i+=1
        new_sequences.append(new_seq)

    # print(f"NMACROS ----- {n_macros}")
    # print(f"best macro {new_macro}")
    # print(new_sequences)
    
    if n_macros-1 > 0:
        best_macros = best_macros + get_macros_from_action_sequences_recur(new_sequences, n_macros=n_macros-1, max_macro_size=max_macro_size, scale_by_len=scale_by_len, action_count=action_count+1)


    # print(f"Best macro has length {len(best_macros[0])} and appears {best_counts[0]} times")
    # print(f"The best macro was {best_macros[0]}")

    return best_macros
    

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

def show_actions(task, action_sequence, name, write_dir=None):
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

    for i, action in enumerate(tqdm(action_sequence)):
        state, _, _ = task.transition(state, action)
        task.set_state(state)
        if write_dir is not None:
            task.visual_file = os.path.join(write_dir, f"state_{i}.png")
        task.render_state(state)
        task.show(ms=500)
        im = imageio.imread(task.visual_file)
        for _ in range(10):
            writer.append_data(im)

    task.set_state(start_state)


def view_policy(Q, task, name):
    _, state_sequence = optimal_policy_actions(Q, task)
    show_states(task, state_sequence, name)

def view_macro(macro, name, task_macros, write_dir=None):
    macro_visualization_task = GridWorldCarry([], (0,0), (5,4), (0,1), board_size=9, action_success_prob=1.0, available_macros=task_macros, macro_visualization=True)
    show_actions(macro_visualization_task, macro, name, write_dir=write_dir)

# def expand_macros_to_primitives(macros, n_prims=10):



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q-Learning on an MDP")
    parser.add_argument("--alpha", type=float, default=0.1, help="The learning rate for the SARSA algorithm")
    parser.add_argument("--gamma", type=float, default=0.8, help="The discount factor for the SARSA algorithm")
    parser.add_argument("--epsilon", type=float, default=0.1, help="The exploration factor for the SARSA algorithm")
    parser.add_argument("--lmbda", type=float, default=0.5, help="Lambda value for SARSA lambda")
    parser.add_argument("--n_episodes", type=int, default=100, help="The number of episodes to train for using the SARSA algorithm")
    parser.add_argument("--save-dir", type=str, default="F:\\Brown\\cs2951x\\trained_models", help="The directory where model weights can be saved to and loaded from")
    parser.add_argument("-l", "--load", action="store_true", help="Load Q from")
    parser.add_argument("-T", "--train", action="store_true", help="Train Q on the task")
    parser.add_argument("-v", "--view", action="store_true", help="View the optimal policy stored in Q")
    parser.add_argument("-m", "--macro", action="store_true", help="Show best macro")
    parser.add_argument('-n', "--name", required=True, help="Name of this experiment")
    args = parser.parse_args()

    # task = GridWorldCarry([(0,0), (1,2), (4,2)], (2,2), (0,3), (3,0), board_size=5)
    task = GridWorldCarry([(0,0), (1,2), (4,2)], (2,2), (7,6), (3,0), board_size=8, available_macros={10: [5,2]})
    # task = GridWorldCarry([(0,0), (1,2), (4,2)], (2,2), (7,6), (3,0), board_size=8, available_macros={10: [5, 2, 0, 5, 2, 0, 5, 0, 4, 5]})
    # task = GridWorldCarry([(0,0), (1,2), (4,2)], (2,2), (7,6), (3,0), board_size=8)

    model_file = os.path.join(args.save_dir, f"{args.name}.npy")

    if args.load:
        Q = np.load(model_file)
    else:
        Q = initialize_Q(task.state_space, task.action_space)

    if args.train:
        all_rewards = []
        for i in range(1):
            print(f"++++++++++++++++++ ROUND {i} ++++++++++++++++++++")
            # Q = initialize_Q(task.state_space, task.action_space)
            Q, rewards = sarsa_lambda(Q, task, args.n_episodes, args.alpha, args.gamma, args.epsilon, args.lmbda)
            np.save(model_file, Q)
            all_rewards.append(rewards)
        log_results(all_rewards, args.name, args.n_episodes)

    if args.view:
        view_policy(Q, task, args.name)

    if args.macro:
        n_seqs = 50
        action_seqs = []
        for _ in range(n_seqs):
            actions, _ = optimal_policy_actions(Q, task)
            # print(len(actions))
            # print(actions)
            action_seqs.append(actions)
        # best_macro = get_macros_from_action_sequences(action_seqs, scale_by_len=True)
        best_macros = get_macros_from_action_sequences_recur(action_seqs, scale_by_len=True, action_count=task.action_space)
        macro_nums = range(task.action_space, task.action_space+len(best_macros))
        macros = {macro_nums[i]:best_macros[i] for i in range(len(best_macros))}
        task.available_macros.update(macros)
        # macros += task.available_macros
    
        for macro in best_macros:
            print(f"Showing macro {macro}")
            view_macro(macro, args.name, task.available_macros)

    
# dec08_05 - 8x8 board no macros
# dec08_06 - 8x8 board top macro (len 10)
# dec08_07 - 8x8 board top macro (len 10)
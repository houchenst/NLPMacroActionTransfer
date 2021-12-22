'''
Run experiments from this script
'''

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from macro_prediction import predict_macros, train_naive_bayes
from macro_utils import expand_macro_set

from mdp import GridWorldCarry
from sarsa import *
import tasks
from macro_prediction import *
from macro_utils import *

def smooth_rewards(mean_rewards, smoothing_factor):
    reward_stack = [np.copy(mean_rewards)]
    for offset in range(1,smoothing_factor):
        new_series = np.copy(mean_rewards)
        new_series[offset:] = mean_rewards[:-offset]
        reward_stack.append(new_series)

    reward_stack = np.stack(reward_stack, axis=-1)
    smoothed_rewards = np.zeros_like(mean_rewards)
    for i in range(mean_rewards.shape[0]):
        curr_slice = reward_stack[i,:]
        epsilon = 0.01
        lower_bound = np.quantile(curr_slice, 0.1) - epsilon
        upper_bound = np.quantile(curr_slice, 0.9) + epsilon
        smoothed_rewards[i] = np.mean(curr_slice[np.logical_and(curr_slice>lower_bound, curr_slice<upper_bound)])
        if i == 4000:
            print(smoothed_rewards[i])
    # smoothed_rewards = np.median(np.stack(reward_stack, axis=-1), axis=-1)
    print("++++++++++")
    return smoothed_rewards

def log_rewards(rewards, filename):
    '''
    Write the rewards to file, append to existing data
    '''

    if os.path.exists(filename):
        existing_data = np.load(filename)
        assert(len(rewards) == existing_data.shape[1])
        new_data = np.vstack((existing_data, rewards))
        np.save(filename, new_data)
    else:
        np.save(filename, np.array([rewards]))

def save_Q(Q, macro_q_path, task_name, macro_type, n_macros):
    path_pattern = os.path.join(macro_q_path, f"{task_name}_{macro_type}_{n_macros}macros_*.npy")
    filenumber = len(glob.glob(path_pattern))
    filename = os.path.join(macro_q_path, f"{task_name}_{macro_type}_{n_macros}macros_{filenumber}.npy")
    np.save(filename, Q)

def train_task_on_macro_set(task_fn, task_name, macro_set, macro_type, n_trials, n_macros_to_add):
    
    for _ in range(n_trials):
        for n in n_macros_to_add:
            task, _ = task_fn()
            task.clear_macros() #unclear why macros are persisting when the task is reinitialized, but they are
            macro_labels = range(10, 10+n)
            macros = macro_set[:n]
            macros_dict = {macro_labels[i]:macros[i] for i in range(len(macro_labels))}
            macros_dict = expand_macro_set(macros_dict) if len(macros_dict) > 0 else macros_dict
            # print(macros_dict)
            task.add_macros(macros_dict)


            Q = initialize_Q(task.state_space, task.action_space)
            print(f"Training on {task_name} using {n} {macro_type} macros")
            Q, rewards = sarsa_lambda(Q, task, args.n_episodes, args.alpha, args.gamma, args.epsilon, args.lmbda, verbose=False)
            rewards_file = os.path.join(rewards_path, f"{task_name}_{macro_type}_{n}macros.npy")
            log_rewards(rewards, rewards_file)
            save_Q(Q, macro_q_path, task_name, macro_type, n)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Run an experiment (train on multiple tasks, extract macros, retrain with macros...)")
    
    # Names 
    parser.add_argument("--expt-name", type=str, required=True, help="The name of this experiment")
    parser.add_argument("--save-dir", type=str, default="F:\\Brown\\cs2951x\\experiments")

    # SARSA Hyperparameters
    parser.add_argument("--alpha", type=float, default=0.1, help="The learning rate for the SARSA algorithm")
    parser.add_argument("--gamma", type=float, default=0.8, help="The discount factor for the SARSA algorithm")
    parser.add_argument("--epsilon", type=float, default=0.1, help="The exploration factor for the SARSA algorithm")
    parser.add_argument("--lmbda", type=float, default=0.5, help="Lambda value for SARSA lambda")
    parser.add_argument("--n_episodes", type=int, default=5000, help="The number of episodes to train for using the SARSA algorithm")

    # Other Hyperparameters
    parser.add_argument("--max-macro-size", type=int, default=10, help="The maximum length of a proposed macro")
    parser.add_argument("--sequences-per-task", type=int, default=20, help="The number of action sequences to generate from each learned policy")
    parser.add_argument("--n-macros", type=int, default=50, help="The total number of macros to extract from the action sequences")

    # Actions
    parser.add_argument("--train", "-T", action="store_true", help="Learn a policy on each task in the training set")
    parser.add_argument("--sequences", "-s", action="store_true", help="Extract successful action sequences from the learned optimal policies")
    parser.add_argument("--macros", "-m", action="store_true", help="Extract the best macros from the successful action sequences")
    parser.add_argument("--predictor", "-p", action="store_true", help="Train a network to predict the optimal macros")
    parser.add_argument("--run-experiments", "-r", action="store_true", help="Run experiments using macros and save results")
    parser.add_argument("--graphs", "-g", action="store_true", help="Show/save graphs comparing rewards with different numbers of macros")
    parser.add_argument("--show", action="store_true", help="Show the graphs that are created")
    parser.add_argument("--all", action="store_true", help="Run all parts of the experiment")

    parser.add_argument("--macro-keyword", type=str, help="Visualize the best macro for the keyword that is passed")
    parser.add_argument("--task-number", type=int, default=1, help="The task to train on or graph")
    args = parser.parse_args()

    if args.all:
        args.train = True
        args.sequences = True
        args.macros = True
        args.predictor = True
        args.run_experiments = True
    
    # ######################  Setup the experiment directory  ######################
    experiment_directory = os.path.join(args.save_dir, args.expt_name)
    if not os.path.exists(experiment_directory):
        os.mkdir(experiment_directory)
    if args.train:
        with open(os.path.join(experiment_directory, "hyperparameters.txt"), "w+") as f:
            f.write(f"Learning Rate (alpha)           -       {args.alpha}\
                      Discount Factor (gamma)         -       {args.gamma}\
                      Exploration Factor (epsilon)    -       {args.epsilon}\
                      Lambda                          -       {args.lmbda}\
                      Number of episodes              -       {args.n_episodes}\
                      Max macro size                  -       {args.max_macro_size}\
                      Sequences per task              -       {args.sequences_per_task}\
                      Number of macros                -       {args.n_macros}\
                ")
            f.close()
    for subdir in ["q_tables", "action_sequences", "macros", "macro_q_tables", "rewards", "graphs", "macro_viz"]:
        subdir_path = os.path.join(experiment_directory, subdir)
        if not os.path.exists(subdir_path):
            os.mkdir(subdir_path)
    

    # ######################  Train on the original tasks without macros  ######################
    if args.train:
        print("+"*20 + " TRAINING " + "+"*20)
        for task_fn, task_name in zip(tasks.all_tasks, tasks.task_names):
            print(f"Training on {task_name}")
            task, caption = task_fn()
            Q = initialize_Q(task.state_space, task.action_space)
            Q, rewards = sarsa_lambda(Q, task, args.n_episodes, args.alpha, args.gamma, args.epsilon, args.lmbda, verbose=False)
            model_file = os.path.join(experiment_directory, "q_tables", f"{task_name}_model.npy")
            np.save(model_file, Q)
    

    # ######################  Extract Action Sequences  ######################
    if args.sequences:
        print("+"*20 + " GENERATING ACTION SEQUENCES " + "+"*20)
        for task_fn, task_name in zip(tasks.all_tasks, tasks.task_names):
            task, caption = task_fn()
            model_file = os.path.join(experiment_directory, "q_tables", f"{task_name}_model.npy")
            Q = np.load(model_file)
            all_action_sequences = []

            for _ in range(args.sequences_per_task):
                actions, _ = optimal_policy_actions(Q, task)
                all_action_sequences.append(actions)
            action_file = os.path.join(experiment_directory, "action_sequences", f"{task_name}_actions.npy")
            np.save(action_file, all_action_sequences)

    # ######################  Extract Action Macros  ######################
    if args.macros:
        print("+"*20 + " FINDING MACROS " + "+"*20)
        all_action_sequences = []
        for task_fn, task_name in zip(tasks.all_tasks, tasks.task_names):
            task, caption = task_fn()
            action_file = os.path.join(experiment_directory, "action_sequences", f"{task_name}_actions.npy")
            action_sequences = list(np.load(action_file, allow_pickle=True))
            # print(action_sequences)
            if task_name in tasks.train_names:
                all_action_sequences += action_sequences

            task_specific_macros = get_macros_from_action_sequences_recur(action_sequences, n_macros=args.n_macros, max_macro_size=args.max_macro_size, scale_by_len=False, action_count=10)

            # express all macros in terms of primitives
            task_specific_macro_file = os.path.join(experiment_directory, "macros", f"{task_name}_macros.npy")
            np.save(task_specific_macro_file, task_specific_macros)
        
        all_macros = get_macros_from_action_sequences_recur(all_action_sequences, n_macros=args.n_macros, max_macro_size=args.max_macro_size, scale_by_len=True, action_count=10)
        # print(all_macros)
        all_macros_file = os.path.join(experiment_directory, "macros", "all_macros.npy")
        np.save(all_macros_file, all_macros)

    if args.predictor:
        print("+"*20 + " TRAINING MACRO PREDICTOR " + "+"*20)
        all_macros_file = os.path.join(experiment_directory, "macros", "all_macros.npy")
        all_macros = list(np.load(all_macros_file, allow_pickle=True))
        macro_labels = range(10, 10+len(all_macros))
        macros_dict = {macro_labels[i]:all_macros[i] for i in range(len(macro_labels))}
        macros_dict = expand_macro_set(macros_dict)
        X,y,word2token,token2word = create_training_data(experiment_directory, macros_dict, tasks.train_tasks, tasks.train_names, tasks.test_tasks)
        model = train_naive_bayes(X,y)

        for task_fn, task_name in zip(tasks.all_tasks, tasks.task_names):
            task, caption = task_fn()
            best_macros, macro_ordering = predict_macros(model, caption, macros_dict, word2token)
            predicted_macro_file = os.path.join(experiment_directory, "macros", f"{task_name}_predicted_macros.npy")
            np.save(predicted_macro_file, best_macros)


    if args.run_experiments:
        print("+"*20 + " RUNNING EXPERIMENTS " + "+"*20)
        macro_q_path = os.path.join(experiment_directory, "macro_q_tables")
        rewards_path = os.path.join(experiment_directory, "rewards")
        graphs_path = os.path.join(experiment_directory, "graph")

        # Number of times to run each experiment
        n_trials = 2
        # Number of macros to try out
        n_macros_to_add = [1, 3, 5, 10, 30, 50]
        # n_macros_to_add = [30]

        task_number = args.task_number
        task_number -=1

        for task_fn, task_name in zip([tasks.all_tasks[task_number]], [tasks.task_names[task_number]]):

            #Train with no macros
            macro_type = "no_macros"
            train_task_on_macro_set(task_fn, task_name, [], macro_type, n_trials, [0])

            # Train on the task specific macros
            macro_type = "task_specific"
            task_specific_macro_file = os.path.join(experiment_directory, "macros", f"{task_name}_macros.npy")
            task_specific_macros = np.load(task_specific_macro_file, allow_pickle=True)
            # 30 macro cutoff for task-specific
            task_specific_n_macros_to_add = [x for x in n_macros_to_add if x < 30]
            # train_task_on_macro_set(task_fn, task_name, task_specific_macros, macro_type, n_trials, task_specific_n_macros_to_add)


            # Train on the full set of macros
            macro_type = "full_set"
            all_macros_file = os.path.join(experiment_directory, "macros", "all_macros.npy")
            all_macros = np.load(all_macros_file, allow_pickle=True)
            train_task_on_macro_set(task_fn, task_name, all_macros, macro_type, n_trials, n_macros_to_add)


            # # Train on the macros that are predicted to be useful
            macro_type = "predicted_useful"
            predicted_macro_file = os.path.join(experiment_directory, "macros", f"{task_name}_predicted_macros.npy")
            predicted_macros = np.load(predicted_macro_file, allow_pickle=True)
            train_task_on_macro_set(task_fn, task_name, predicted_macros, macro_type, n_trials, n_macros_to_add)


    if args.graphs:
        graph_dir = os.path.join(experiment_directory, "graphs")
        n_macros_to_add = [1, 3, 5, 10, 30,50]
        task_number = args.task_number
        task_number -= 1
        for task_fn, task_name in zip([tasks.all_tasks[task_number]], [tasks.task_names[task_number]]):
            for macro_type in ["task_specific", "full_set", "predicted_useful"]:
                f = plt.figure()
                ax = f.add_subplot(111)
                plt.yscale("symlog")

                # show no macros
                no_macro_path = os.path.join(experiment_directory, "rewards", f"{task_name}_no_macros_0macros.npy")
                if os.path.exists(no_macro_path):
                    rewards = np.load(no_macro_path)
                    # convert to timesteps
                    rewards = rewards - 100.
                    cutoff = np.ones_like(rewards) * -100000.
                    rewards = np.max(np.stack([rewards, cutoff], axis=-1), axis=-1)

                    mean_rewards = np.mean(rewards, axis=0)
                    smoothing_factor = 50
                    mean_rewards = smooth_rewards(mean_rewards, smoothing_factor)

                    xs = np.arange(rewards.shape[1])
                    ax.plot(xs, mean_rewards, label="No macros")

                for n in n_macros_to_add:
                    reward_path = os.path.join(experiment_directory, "rewards", f"{task_name}_{macro_type}_{n}macros.npy")
                    if not os.path.exists(reward_path):
                        break
                    rewards = np.load(reward_path)

                    # convert to timesteps
                    rewards = rewards - 100.
                    cutoff = np.ones_like(rewards) * -100000.
                    rewards = np.max(np.stack([rewards, cutoff], axis=-1), axis=-1)
                    mean_rewards = np.mean(rewards, axis=0)
                    smoothing_factor = 50
                    # print(f"{n} macros   : {np.max(rewards)}")
                    mean_rewards = smooth_rewards(mean_rewards, smoothing_factor)

                    xs = np.arange(rewards.shape[1])
                    ax.plot(xs, mean_rewards, label=f"{n} macros")
                ax.legend()
                if macro_type == "task_specific":
                    f.suptitle("Number of Time Steps to Reward Using Task-Specific Macros")
                if macro_type == "full_set":
                    f.suptitle("Number of Time Steps to Reward Using All Macros")
                if macro_type == "predicted_useful":
                    f.suptitle("Number of Time Steps to Reward Using Macros Suggested From Caption")
                ax.set_ylabel("Number of time steps to reward")
                ax.set_xlabel("Episodes")
                graph_path = os.path.join(graph_dir, f"{task_name}_{macro_type}.png")
                plt.savefig(graph_path, dpi=200)
                if args.show:
                    plt.show()

        optimal_n_macros = 5
        for task_fn, task_name in zip([tasks.all_tasks[task_number]], [tasks.task_names[task_number]]):
            f = plt.figure()
            ax = f.add_subplot(111)
            plt.yscale("symlog")
            no_macro_path = os.path.join(experiment_directory, "rewards", f"{task_name}_no_macros_0macros.npy")
            if os.path.exists(no_macro_path):
                rewards = np.load(no_macro_path)
                # convert to timesteps
                rewards = rewards - 100.
                cutoff = np.ones_like(rewards) * -100000.
                rewards = np.max(np.stack([rewards, cutoff], axis=-1), axis=-1)

                mean_rewards = np.mean(rewards, axis=0)
                smoothing_factor = 50
                mean_rewards = smooth_rewards(mean_rewards, smoothing_factor)

                xs = np.arange(rewards.shape[1])
                ax.plot(xs, mean_rewards, label="No macros")
            for macro_type in ["full_set", "predicted_useful", "task_specific"]:
                reward_path = os.path.join(experiment_directory, "rewards", f"{task_name}_{macro_type}_{optimal_n_macros}macros.npy")
                if not os.path.exists(reward_path):
                        continue
                rewards = np.load(reward_path)

                # convert to timesteps
                rewards = rewards - 100.
                cutoff = np.ones_like(rewards) * -100000.
                rewards = np.max(np.stack([rewards, cutoff], axis=-1), axis=-1)
                mean_rewards = np.mean(rewards, axis=0)
                smoothing_factor = 50
                # print(f"{n} macros   : {np.max(rewards)}")
                mean_rewards = smooth_rewards(mean_rewards, smoothing_factor)
                
                xs = np.arange(rewards.shape[1])
                macro_name = ""
                if macro_type == "task_specific":
                    macro_name = "Task-specific"
                if macro_type == "full_set":
                    macro_name = "All"
                if macro_type == "predicted_useful":
                    macro_name = "Model-predicted"
                ax.plot(xs, mean_rewards, label=f"{macro_name} macros")
            ax.legend()
            f.suptitle(f"Number of Time Steps to Reward Using {optimal_n_macros} Macros")
            ax.set_ylabel("Number of time steps to reward")
            ax.set_xlabel("Episodes")
            graph_path = os.path.join(graph_dir, f"{task_name}_{optimal_n_macros}macros.png")
            plt.savefig(graph_path, dpi=200)
            if args.show:
                plt.show()

    if args.macro_keyword:
        print("+"*20 + " VISUALIZING BEST MACROS FOR KEYWORD: " + args.macro_keyword + "+"*20)
        all_macros_file = os.path.join(experiment_directory, "macros", "all_macros.npy")
        all_macros = list(np.load(all_macros_file, allow_pickle=True))
        macro_labels = range(10, 10+len(all_macros))
        macros_dict = {macro_labels[i]:all_macros[i] for i in range(len(macro_labels))}
        macros_dict = expand_macro_set(macros_dict)
        X,y,word2token,token2word = create_training_data(experiment_directory, macros_dict, tasks.train_tasks, tasks.train_names, tasks.test_tasks)
        model = train_naive_bayes(X,y)

        best_macros, macro_ordering = predict_macros(model, args.macro_keyword, macros_dict, word2token)
        print(best_macros)

        macro_number = 0
        macro_viz_path = os.path.join(experiment_directory, "macro_viz", f"macro_keyword_{macro_number}")
        os.mkdir(macro_viz_path)
        view_macro(list(best_macros[14]), args.macro_keyword, {}, write_dir=macro_viz_path)

        # for task_fn, task_name in zip(tasks.all_tasks, tasks.task_names):
        #     task, caption = task_fn()
        #     best_macros, macro_ordering = predict_macros(model, caption, macros_dict, word2token)
        #     predicted_macro_file = os.path.join(experiment_directory, "macros", f"{task_name}_predicted_macros.npy")
        #     np.save(predicted_macro_file, best_macros)


    


    

    

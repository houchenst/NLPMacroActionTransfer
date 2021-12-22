# Using Task Descriptions for Macro Action Transfer

A method for proposing macro actions in the form of action sequences and identifying the most useful macros using natural language task descriptions.

### Structure
- `main.py` - used to run experiments.
- `mdp.py` - defines a class of MDP used for this project.
- `tasks.py` sets up specific tasks and descriptions.
- `sarsa.py` - implements SARSA lambda.
- `macro_prediction.py` - trains a Naive Bayes Classifier to predict useful macros.

### Running The Code
`python main.py --expt-name <experiment name> --save-dir <save directory> --all`

This will learn a policy on all tasks laid out in `tasks.py`, extract action sequences from these policies, propose macros from these action sequences, train a Naive Bayes classifier using the macros and task descriptions, and finally re-train the macros. All intermediate results including Q tables, rewards, action sequences, and macros will be written to the save directory and can be loaded later. Instead of passing `--all` you can pass other flags that only run parts of this process. Use `--help` for more information.

`python main.py --expt-name <experiment name> --save-dir <save directory> --graphs --show --task-number <task number>`

This will show the reward curves for a specific task using different macro sets. The task number is an index into `all_tasks` which is defined in `tasks.py`. The passed task number is for re-training and graphing, since it would be slow to do either of these on all of the tasks.

`python main.py --expt-name <experiment name> --save-dir <save directory> --macro-keyword <word>`

Displays the top macro predicted by the Naive Bayes classifier for this word. The word must be present in at least one of the task descriptions.


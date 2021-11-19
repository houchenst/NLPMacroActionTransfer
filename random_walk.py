from mdp import GridWorldCarry
from numpy.random import randint
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    task = GridWorldCarry([(0,0), (1,2), (5,6), (4,2)], (2,2), (0,3), (3,0), board_size=7)

    parser=argparse.ArgumentParser(description="Test the transition function of the mdp under different actions and states to make sure there are no errors")
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--random-walk", action="store_true", help="performs a random walk through the state space, intermittently showing results and stopping at 10,000 steps or the goal state")
    options.add_argument("--exhaustive-trial", action="store_true", help="runs the transition function for every state, action pairing, to ensure that they all work")
    args = parser.parse_args()


    if args.random_walk:
        max_actions = 10000
        i=0
        while i < max_actions:
            action = randint(0, 10)
            new_state_number = task.transition(task.state_number, action)
            assert(task.is_valid_state(new_state_number))
            task.set_state(new_state_number)
            task.render_state(task.state_number)
            if i%1000 == 0:
                print(f"Showing state {i}")
                task.show(ms=1000)
            elif i%100 ==0:
                print(i)
            reward = task.reward(task.state_number)
            if reward > 0.:
                print(f"REWARD: {reward}")
                task.show(ms=10000)
                break
            i+=1
    if args.exhaustive_trial:
        n_states = task.board_size**4 * 32
        n_actions = 10
        invalid_states = 0
        valid_states = 0
        for sn in tqdm(range(n_states)):
            if task.is_valid_state(sn):
                valid_states += 1
                for a in range(n_actions):
                    task.set_state(sn)
                    new_state_number = task.transition(sn, a)
                    if not task.is_valid_state(new_state_number):
                        task.render_state(new_state_number)
                        assert(task.is_valid_state(new_state_number))
            else:
                invalid_states += 1
        print(f"{invalid_states/(invalid_states+valid_states)*100.:.2f}% of states were invalid")
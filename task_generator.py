'''
Procedurally generate tasks and captions in the same state/action space
'''

from mdp import GridWorldCarry

def gen_task_and_caption(world_size, task_type, agent_loc, target_loc, object_loc, obstacles, n_obstacles):
    '''
    world_size - the dimensions of each edge of the gridworld (always square)
    task_type - one of ["CARRY_OBJ", "REACH_GOAL", ] which describes the goal of the task
    agent_loc - the starting location of the agent, subject to noise
    target_loc - the starting location of the target, subject to noise
    object_loc - the starting location of the object, subject to noise
    obstacles - a list of object distribution centers
    max_obstacles - the total number of obstacles to place in the scene
    '''

    # TODO: build a reward function depending on the type of task


    # TODO: Randomly choose agent location according to Gaussian distribution around the agent_loc


    #TODO: Same procedure as for agent, but for target and object


    #TODO: Build probability distribution across the Grid according to obstacle distribution centers


    #TODO: sample n obstacles from the obstacle distribution probability



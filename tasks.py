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

def far_or_near(agent_location, object_location):
    '''
    Returns the string "far away" or "nearby" depending on how close the agent is to the object.
    '''
    #use a manhattan distance since that is how the agent has to move
    distance = sum([abs(agent_location[i]-object_location[i]) for i in range(len(agent_location))])
    far_threshold = 5
    if distance >= far_threshold:
        return "far away"
    else:
        return "nearby"

def task1():
    agent_location = (7,6)
    object_location = (2,2)
    target_location = (3,0)
    obstacle_locations = [(0,0), (1,2), (4,2)]
    task = GridWorldCarry(obstacle_locations, object_location, agent_location, target_location, board_size=8)

    caption = f"Move west then south to the object. Carry the object south then east to the target. The object is {far_or_near(agent_location, object_location)}."
    return task, caption

def task2():
    agent_location = (2,2)
    object_location = (7,2)
    target_location = (7,6)
    obstacle_locations = [(5,5), (4,5), (3,5), (5,6), (4,1)]
    task = GridWorldCarry(obstacle_locations, object_location, agent_location, target_location, board_size=8)

    caption = f"Move east towards the object. Carry the object north to the target. The object is {far_or_near(agent_location, object_location)}."
    return task, caption

def task3():
    agent_location = (1,6)
    object_location = (1,1)
    target_location = (6,6)
    obstacle_locations = [(2,7),(2,6),(2,5),(2,4),(2,3),(3,3),(4,3),(5,3),(5,4),(5,5),(5,6),(5,7)]
    task = GridWorldCarry(obstacle_locations, object_location, agent_location, target_location, board_size=8)

    caption = f"Move south towards the object. Carry the object east then north to the target. The object is {far_or_near(agent_location, object_location)}."
    return task, caption

def task4():
    # single sharp turn
    agent_location = (6,1)
    object_location = (2,5)
    target_location = (0,5)
    obstacle_locations = [(0,4),(1,4),(2,4),(3,4),(4,4),(5,5),(4,7),(3,7),(5,4),(5,7),(6,7),(7,7),(7,6),(7,5),(3,6),(7,5),(7,4)]
    task = GridWorldCarry(obstacle_locations, object_location, agent_location, target_location, board_size=8)

    caption = f"Move north towards the object. Make a sharp turn. Carry the object to the target. The object is {far_or_near(agent_location, object_location)}."
    return task, caption

def task5():
    #straight north
    agent_location = (3,0)
    object_location = (3,4)
    target_location = (3,7)
    obstacle_locations = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7)]
    task = GridWorldCarry(obstacle_locations, object_location, agent_location, target_location, board_size=8)

    caption = f"Move north towards the object. Carry the object north to the target. The object is {far_or_near(agent_location, object_location)}."
    return task, caption

def task7():
    #zigzag
    agent_location = (0,1)
    object_location = (6,4)
    target_location = (7,4)
    obstacle_locations = [(0,0),(1,0),(2,0),(3,0),(0,4),(0,5),(0,6),(0,7),(1,6),(2,6),(3,1),(3,2),(0,2),(1,2),(0,3),(3,3),(3,4),(2,4),(7,5),(7,6),(7,7),(4,5),(4,6),(6,7),(6,6)]
    task = GridWorldCarry(obstacle_locations, object_location, agent_location, target_location, board_size=8)

    caption = f"Make a sharp turn multiple times. Carry the object to the target. The object is {far_or_near(agent_location, object_location)}."
    return task, caption

def task8():
    #spiral
    agent_location = (4,4)
    object_location = (5,6)
    target_location = (1,6)
    obstacle_locations = [(4,5),(5,5),(5,4),(5,3),(4,3),(3,5),(2,5),(2,4),(2,3),(3,1),(2,2),(2,1),(3,1),(4,1),(5,1),(6,1),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(0,3),(0,4),(0,5)]
    task = GridWorldCarry(obstacle_locations, object_location, agent_location, target_location, board_size=8)

    caption = f"Move in a spiral. Carry the object to the target. The object is {far_or_near(agent_location, object_location)}."
    return task, caption

def task10():
    #double board traverse
    agent_location = (1,0)
    object_location = (5,6)
    target_location = (0,1)
    obstacle_locations = []
    task = GridWorldCarry(obstacle_locations, object_location, agent_location, target_location, board_size=8)

    caption = f"Move north and east towards the object. Carry the object south and west to the target. The object is {far_or_near(agent_location, object_location)}."
    return task, caption

def task11():
    #several obstacles
    agent_location = (5,2)
    object_location = (4,6)
    target_location = (1,3)
    obstacle_locations = [(2,0),(2,5),(3,4),(5,6),(4,7),(1,0),(4,4),(3,0),(3,1)]
    task = GridWorldCarry(obstacle_locations, object_location, agent_location, target_location, board_size=8)

    caption = f"Move north towards the object. Carry the object west, then south to the target. The object is {far_or_near(agent_location, object_location)}."
    return task, caption

def task9():
    #one sharp turn
    agent_location = (1,4)
    object_location = (5,6)
    target_location = (7,6)
    obstacle_locations = [(2,0),(2,1),(2,2),(1,3),(2,5),(2,6),(2,7),(3,0),(3,1),(3,2),(3,4),(3,5),(3,6),(3,7),(4,0),(4,1),(4,2),(5,3),(4,5),(4,6),(4,7)]
    task = GridWorldCarry(obstacle_locations, object_location, agent_location, target_location, board_size=8)

    caption = f"Move east, making a sharp turn, then head north towards the object. Carry the object to the target. The object is {far_or_near(agent_location, object_location)}."
    return task, caption

def task6():
    #straight south
    agent_location = (3,7)
    object_location = (3,4)
    target_location = (3,0)
    obstacle_locations = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7)]
    task = GridWorldCarry(obstacle_locations, object_location, agent_location, target_location, board_size=8)

    caption = f"Move south towards the object. Carry the object south to the target. The object is {far_or_near(agent_location, object_location)}."
    return task, caption

all_tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10, task11]
task_names = [f"task{i+1}" for i in range(len(all_tasks))]

train_set = [1,2,3,4,5,7,8,10,11]
train_tasks = [all_tasks[i-1] for i in train_set]
train_names = [task_names[i-1] for i in train_set]

test_set = [6,9]
test_tasks = [all_tasks[i-1] for i in test_set]
test_names = [task_names[i-1] for i in test_set]

if __name__ == "__main__":
    specific_task = None
    # specific_task = -1
    all_tasks = [task1(), task2(), task3(), task4(), task5(), task6(),task7(),task8(),task9(),task10(),task11()]
    if specific_task is not None:
        all_tasks = [all_tasks[specific_task]]
    for i, (task, caption) in enumerate(all_tasks):
        task.render_state(task.state_number)
        print(f"Task {i+1} - {caption}")
        showtime = 2000 if len(all_tasks) > 1 else 10000
        task.show(ms=showtime)

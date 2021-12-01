'''
Defines a class of Markov Decision Processes
'''

import drawSvg as draw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.random.mtrand import poisson
import math


class GridWorldCarry():
    '''
    This task is similar to GridWorld, except it involves carrying/pushing an object to a target    
    '''

    def __init__(self, obstacles, object_location, agent_location, goal_location, board_size=7, visual_file="F:\Brown\cs2951x\stateview.png", action_success_prob=0.95):
        '''
        obstacles       - tuple indices of obstacles on the board
        object_location - the location of the object on the board
        agent_location  - the location of the agent on the board
        board_size      - board will have dimensions (board_size x board_size)
        '''
        #check that the agent isn't on an obstacle or outside the map
        assert(agent_location not in obstacles)
        assert(agent_location[0] >= 0 and agent_location[0] < board_size)
        assert(agent_location[1] >= 0 and agent_location[1] < board_size)
        # check that the object isn't on an obstacle or outside the map
        assert(object_location not in obstacles)
        assert(object_location[0] >= 0 and object_location[0] < board_size)
        assert(object_location[1] >= 0 and object_location[1] < board_size)
        # check that the goal isn't on an obstacle or outside the map
        assert(goal_location not in obstacles)
        assert(goal_location[0] >= 0 and goal_location[0] < board_size)
        assert(goal_location[1] >= 0 and goal_location[1] < board_size)

        self.obstacles = set(obstacles)
        self.goal_location = goal_location
        self.agent_location = agent_location
        self.object_location = object_location
        self.board_size = board_size
        self.agent_orientation = 0
        self.forelimbs_position = 0
        self.rear_left_position = 0
        self.rear_right_position = 0
        self.state_number = self.get_state_number(self.agent_location, self.object_location, self.agent_orientation, self.forelimbs_position, self.rear_left_position, self.rear_right_position)
        self.visual_file = visual_file

        self.state_space = self.board_size**4 * 32
        self.action_space = 10
        #Probability of a stochastic action succeeding
        self.action_success_prob = action_success_prob


    def get_state_number(self, agent_location, object_location, agent_orientation, forelimbs, rear_left, rear_right):
        '''
        Returns the integer state number
        '''
        state_number = rear_right
        state_number += rear_left * 2
        state_number += forelimbs * 2 * 2
        state_number += agent_orientation * 2 * 2 * 2
        state_number += object_location[0] * 2 * 2 * 2 * 4
        state_number += object_location[1] * 2 * 2 * 2 * 4 * self.board_size
        state_number += agent_location[0] * 2 * 2 * 2 * 4 * self.board_size * self.board_size
        state_number += agent_location[1] * 2 * 2 * 2 * 4 * self.board_size * self.board_size * self.board_size

        return state_number

    def get_state_components(self, state_number):
        '''
        Decomposes a state number into useable aspects of the state
        '''
        rear_right = state_number%2
        rear_left = state_number//(2)%2
        forelimbs = state_number//(2*2)%2
        agent_orientation = state_number//(2*2*2)%4
        object_location = (state_number//(2*2*2*4)%self.board_size, state_number//(2*2*2*4*self.board_size)%self.board_size)
        agent_location = (state_number//(2*2*2*4*self.board_size*self.board_size)%self.board_size, state_number//(2*2*2*4*self.board_size*self.board_size*self.board_size)%self.board_size)

        return agent_location, object_location, agent_orientation, forelimbs, rear_left, rear_right

    def set_state(self, state_number):
        agent_location, object_location, agent_orientation, forelimbs, rear_left, rear_right = self.get_state_components(state_number)
        self.set_state_components(agent_location, object_location, agent_orientation, forelimbs, rear_left, rear_right)
        self.state_number = state_number

    def set_state_components(self, agent_location, object_location, agent_orientation, forelimbs, rear_left, rear_right):
        '''
        Decomposes a state number into useable aspects of the state
        '''
        self.agent_location = agent_location
        self.object_location = object_location
        self.agent_orientation = agent_orientation
        self.forelimbs_position = forelimbs
        self.rear_left_position = rear_left
        self.rear_right_position = rear_right

    def transition(self, state_number, action):
        '''
        Returns a new state number that is transitioned to after taking the provided action in the provided state number
        '''
        agent_location, object_location, agent_orientation, forelimbs, rear_left, rear_right = self.get_state_components(state_number)
        # print(f"Transition agent location: {agent_location}")

        def move_agent_and_object(agent_location, object_location, forelimbs_position, orientation, movement_direction, stochasticity=0.9):
            '''
            A subroutine for transitioning the agent and object positions which considers the movement direction, obstacles, success probabilitly, and whether the object is grasped
            '''
            action_success = np.random.rand() < stochasticity
            
            # If the stochastic action is unsuccessful, return the same positions
            if not action_success:
                return agent_location, object_location
            
            # If the stochastic action is successful, try to move the agent/object
            new_agent_location = self.move_position(agent_location, movement_direction)
            if new_agent_location is None:
                return agent_location, object_location

            # Case where object is grasped or pushed
            object_is_grasped = self.object_grasped(object_location, agent_location, forelimbs_position, orientation)
            new_forelimbs_location = self.move_position(new_agent_location, orientation)
            object_is_pushed = new_agent_location == object_location or (forelimbs and new_forelimbs_location == object_location)
            if object_is_grasped or object_is_pushed:
                new_object_location = self.move_position(object_location, movement_direction)
                if new_agent_location is None or new_object_location is None:
                    return agent_location, object_location
                for pos in [new_agent_location, new_object_location]:
                    if pos in self.obstacles:
                        return agent_location, object_location
                return new_agent_location, new_object_location

            # Object is not grabbed or pushed
            else:                
                if new_agent_location is None or new_agent_location in self.obstacles:
                    return agent_location, object_location
                else:
                    return new_agent_location, object_location
        
        def arm_and_push_location(agent_location, orientation, cw):
            '''
            Returns the square that the agent's arms will occupy and the square that the object would be pushed to if it were hit by the arms
            '''
            push_direction = (orientation - 2) % 4
            arms_direction = (orientation + cw - (not cw)) % 4
            arms_location = self.move_position(agent_location, arms_direction)
            if arms_location is None:
                return None, None
            else:
                return arms_location, self.move_position(arms_location, push_direction)       
        
        def rotate_agent_and_object(agent_location, object_location, forelimbs_position, orientation, cw):
            '''
            A subroutine for doing an agent rotation transition. Checks whether the forelimbs collide with an object and then moves the object if it has collided.
            '''
            # find the orientation if we are able to rotate
            agent_orientation = (orientation + cw - (not cw)) % 4
            arms_location, push_location = arm_and_push_location(agent_location, orientation, cw)
            if object_location == arms_location and forelimbs_position:
                if push_location in self.obstacles or push_location is None:
                    # in this case the roatation is blocked
                    agent_orientation = orientation
                else:
                    object_location = push_location
            if self.object_grasped(object_location, agent_location, forelimbs_position, orientation):
                if arms_location in self.obstacles or arms_location is None:
                    agent_orientation = orientation
                else:
                    object_location = arms_location
            
            return agent_orientation, object_location

        def forelimb_grasp(agent_location, agent_orientation, object_location, stochasticity=0.9):
            '''
            If the agent grasps, see whether the object is pushed
            '''
            action_success = np.random.rand() < stochasticity
            if not action_success:
                grasp_location = self.move_position(agent_location, agent_orientation)
                if grasp_location is not None:
                    push_location = self.move_position(grasp_location, agent_orientation)
                    if grasp_location == object_location and push_location not in self.obstacles and push_location is not None:
                        return push_location
            return object_location


        # rear left leg up (DETERMINISTIC)
        if action == 0:
            rear_left = 1
        # rear left leg down (DETERMINISTIC)
        if action == 1:
            rear_left = 0
        #rear right leg up (DETERMINISTIC)
        if action == 2:
            rear_right = 1
        #rear right leg down (DETERMINISTIC)
        if action == 3:
            rear_right = 0
        #rear legs both up (STOCHASTIC)
        if action == 4:
            if not rear_left and not rear_right:
                    agent_location, object_location = move_agent_and_object(agent_location, object_location, forelimbs, agent_orientation, (agent_orientation-2)%4, stochasticity=self.action_success_prob)
            rear_left = 1
            rear_right = 1
        #rear legs both down (STOCHASTIC)
        if action == 5:
            if rear_left and rear_right:
                agent_location, object_location = move_agent_and_object(agent_location, object_location, forelimbs, agent_orientation, agent_orientation, stochasticity=self.action_success_prob)
            rear_left = 0
            rear_right = 0
        #rear legs both CW (STOCHASTIC)
        if action == 6:
            if rear_left == 0 and rear_right == 1:
                agent_orientation, object_location = rotate_agent_and_object(agent_location, object_location, forelimbs, agent_orientation, False)
            rear_left = 1
            rear_right = 0
        #rear legs both CCW (STOCHASTIC)
        if action == 7:
            if rear_left == 1 and rear_right == 0:
                agent_orientation, object_location = rotate_agent_and_object(agent_location, object_location, forelimbs, agent_orientation, True)
            rear_left = 0
            rear_right = 1
        #forelimbs up (STOCHASTIC)
        if action == 8:
            forelimbs = 1
            object_location = forelimb_grasp(agent_location, agent_orientation, object_location, stochasticity=self.action_success_prob)
        #forelimbs down (DETERMINISTIC)
        if action == 9:
            forelimbs = 0

        new_state_number = self.get_state_number(agent_location, object_location, agent_orientation, forelimbs, rear_left, rear_right)
        return new_state_number

    def reward(self, state):
        '''
        The reward here is just defined over state (having the object on the goal square achieves the reward)
        '''
        _, object_location, _, _, _, _ = self.get_state_components(state)
        if object_location == self.goal_location:
            return 100.
        else:
            return -1.0

    def object_grasped(self, object_location, agent_location, forelimbs_position, orientation):
        if forelimbs_position == 0:
            return False
        else:
            grasp_position = self.move_position(agent_location, orientation)
            return object_location == grasp_position

    def move_position(self, position, direction):
        '''
        Takes in a position and a cardinal direction and outputs the position that is one step in that direction.
        Returns None if the new position is not on the board
        '''
        new_x = (position[0]-1 if direction == 3 else position[0]+1) if direction % 2 else position[0]
        new_y = (position[1]-1 if direction == 2 else position[1]+1) if not direction % 2 else position[1]
        if new_x < 0 or new_y < 0 or new_x >= self.board_size or new_y >= self.board_size:
            return None
        else:
            return (new_x, new_y)

    @staticmethod
    def limb_bases_and_directions(orientation, agent_location, forelimbs, rear_left, rear_right, cell_size):
        '''
        Returns the coordinates (in image space) of the two endpoints of each leg
        '''
        limb_endpoints = []

        i,j = agent_location
        cell_bottom_left = np.array([i*cell_size+1, j*cell_size+1])
        agent_corners = [cell_bottom_left + np.array([0.2*cell_size, 0.2*cell_size]),
                         cell_bottom_left + np.array([0.8*cell_size, 0.2*cell_size]),
                         cell_bottom_left + np.array([0.8*cell_size, 0.8*cell_size]),
                         cell_bottom_left + np.array([0.2*cell_size, 0.8*cell_size])]

        # corner directions (outwards - 45 degrees, outward, and outward+45 degrees) for each corner
        corner_directions = [np.array([[0.0, -1.0], [-math.sqrt(2.0)/2.0, -math.sqrt(2.0)/2.0], [-1.0, 0.0]]),
                            np.array([[1.0, 0.0], [math.sqrt(2.0)/2.0, -math.sqrt(2.0)/2.0], [0.0, -1.0]]),
                            np.array([[0.0, 1.0], [math.sqrt(2.0)/2.0, math.sqrt(2.0)/2.0], [1.0, 0.0]]),
                            np.array([[-1.0, 0.0], [-math.sqrt(2.0)/2.0, math.sqrt(2.0)/2.0], [0.0, 1.0]]),]

        corner_directions = [x*cell_size*0.4 for x in corner_directions]

        # adjust for orientation before assigning corners
        for _ in range(orientation):
            agent_corners = [agent_corners.pop()] + agent_corners
            corner_directions = [corner_directions.pop()] + corner_directions
        
        rear_left_start, rear_right_start, front_right_start, front_left_start = agent_corners
        rear_left_dir, rear_right_dir, front_right_dir, front_left_dir = corner_directions

        # choose the correct direction based on limb positioning
        front_left_dir = front_left_dir[forelimbs+1]
        front_right_dir = front_right_dir[int(not forelimbs)]
        rear_right_dir = rear_right_dir[int(not rear_right)]
        rear_left_dir = rear_left_dir[rear_left + 1]

        # return start,end for each limb in cw order (start at top left)
        endpoints = [(tuple(front_left_start.astype(int)), tuple((front_left_start+front_left_dir).astype(int))),
                    (tuple(front_right_start.astype(int)), tuple((front_right_start+front_right_dir).astype(int))),
                    (tuple(rear_right_start.astype(int)), tuple((rear_right_start+rear_right_dir).astype(int))),
                    (tuple(rear_left_start.astype(int)), tuple((rear_left_start+rear_left_dir).astype(int)))]
        
        return endpoints

    def is_valid_state(self, state_number):
        '''
        Checks whether the provided state number is a valid configuration
        '''
        agent_location, object_location, _, _, _, _ = self.get_state_components(state_number)

        # Error if the agent or object have invalid coordinates
        assert(agent_location[0] >= 0 and agent_location[0]<self.board_size and agent_location[1] >= 0 and agent_location[1]<self.board_size)
        assert(object_location[0] >= 0 and object_location[0]<self.board_size and object_location[1] >= 0 and object_location[1]<self.board_size)

        # State is invalid if agent or object is on top of an obstacle
        if agent_location in self.obstacles:
            return False
        if object_location in self.obstacles:
            return False

        # State is invalid if the agent is on top of the object
        if agent_location == object_location:
            return False
        
        return True


    def render_state(self, state):

        agent_location, object_location, agent_orientation, forelimbs, rear_left, rear_right = self.get_state_components(state)

        cell_size = 50

        d = draw.Drawing(self.board_size*cell_size+2, self.board_size*cell_size+2, displayInline=False)
        r = draw.Rectangle(0, 0, self.board_size*cell_size+2, self.board_size*cell_size+2, stroke_width=1, stroke='black', fill='#fff')
        d.append(r)

        for i in range(self.board_size):
            for j in range(self.board_size):
                # see if we want an empty or filled square
                if not (i,j) in self.obstacles:
                    r = draw.Rectangle(i*cell_size+1,j*cell_size+1,cell_size,cell_size, stroke_width=1, stroke='black', fill='#fff')
                    d.append(r)
                else:
                    r = draw.Rectangle(i*cell_size+1,j*cell_size+1,cell_size,cell_size, stroke_width=1, stroke='black', fill='black')
                    d.append(r)

                # see if we should display the target location
                if (i,j) == self.goal_location:
                    r = draw.Rectangle((i+0.2)*cell_size+1,(j+0.2)*cell_size+1,0.6*cell_size,0.6*cell_size, stroke_width=4, stroke='green', fill='#fff')
                    d.append(r)
                    r = draw.Rectangle((i+0.35)*cell_size+1,(j+0.35)*cell_size+1,0.3*cell_size,0.3*cell_size, fill='green')
                    d.append(r)

                # see if we should display the object
                if (i,j) == object_location:
                    c = draw.Circle((i+0.5)*cell_size+1, (j+0.5)*cell_size+1, cell_size/2-4, fill="red")
                    d.append(c)

                # see if we should display the agent
                if (i,j) == agent_location:
                    # display the body
                    r = draw.Rectangle((i+0.2)*cell_size+1,(j+0.2)*cell_size+1,0.6*cell_size,0.6*cell_size, stroke_width=4, stroke='purple', fill='purple')
                    d.append(r)

                    # display the orientation
                    # calculate the midpoints on each side of the agent's body
                    n_mid = ((i+0.5)*cell_size+1, (j+0.8)*cell_size+1)
                    e_mid = ((i+0.8)*cell_size+1, (j+0.5)*cell_size+1)
                    s_mid = ((i+0.5)*cell_size+1, (j+0.2)*cell_size+1)
                    w_mid = ((i+0.2)*cell_size+1, (j+0.5)*cell_size+1)
                    # North
                    if agent_orientation == 0:
                        l = draw.Lines(w_mid[0], w_mid[1], n_mid[0], n_mid[1], e_mid[0], e_mid[1], close=True, fill='#eeee00', stroke='#eeee00')
                        d.append(l)
                    #East
                    if agent_orientation == 1:
                        l = draw.Lines(n_mid[0], n_mid[1], e_mid[0], e_mid[1], s_mid[0], s_mid[1], close=True, fill='#eeee00', stroke='#eeee00')
                        d.append(l)
                    #South
                    if agent_orientation == 2:
                        l = draw.Lines(e_mid[0], e_mid[1], s_mid[0], s_mid[1], w_mid[0], w_mid[1], close=True, fill='#eeee00', stroke='#eeee00')
                        d.append(l)
                    #West
                    if agent_orientation == 3:
                        l = draw.Lines(s_mid[0], s_mid[1], w_mid[0], w_mid[1], n_mid[0], n_mid[1], close=True, fill='#eeee00', stroke='#eeee00')
                        d.append(l)

                # r = draw.Rectangle(-10,-10,20,20, fill='#1248ff')
                # r.appendTitle("Our first rectangle")  # Add a tooltip
                # d.append(r)

        limbs = self.limb_bases_and_directions(agent_orientation, agent_location, forelimbs, rear_left, rear_right, cell_size)
        for limb in limbs:
            start, end = limb
            l = draw.Lines(start[0], start[1], end[0], end[1], close=False, stroke="black")
            d.append(l)

        # Draw a rectangle

        # # Draw a circle
        # d.append(draw.Circle(-40, -10, 30,
        #             fill='red', stroke_width=2, stroke='black'))

        # # Draw an arbitrary path (a triangle in this case)
        # p = draw.Path(stroke_width=2, stroke='lime',
        #             fill='black', fill_opacity=0.2)
        # p.M(-10, 20)  # Start path at point (-10, 20)
        # p.C(30, -10, 30, 50, 70, 20)  # Draw a curve to (70, 20)
        # d.append(p)

        # # Draw text
        # d.append(draw.Text('Basic text', 8, -10, 35, fill='blue'))  # Text with font size 8
        # d.append(draw.Text('Path text', 8, path=p, text_anchor='start', valign='middle'))
        # d.append(draw.Text(['Multi-line', 'text'], 8, path=p, text_anchor='end'))

        # # Draw multiple circular arcs
        # d.append(draw.ArcLine(60,-20,20,60,270,
        #             stroke='red', stroke_width=5, fill='red', fill_opacity=0.2))
        # d.append(draw.Arc(60,-20,20,60,270,cw=False,
        #             stroke='green', stroke_width=3, fill='none'))
        # d.append(draw.Arc(60,-20,20,270,60,cw=True,
        #             stroke='blue', stroke_width=1, fill='black', fill_opacity=0.3))

        # # Draw arrows
        # arrow = draw.Marker(-0.1, -0.5, 0.9, 0.5, scale=4, orient='auto')
        # arrow.append(draw.Lines(-0.1, -0.5, -0.1, 0.5, 0.9, 0, fill='red', close=True))
        # p = draw.Path(stroke='red', stroke_width=2, fill='none',
        #             marker_end=arrow)  # Add an arrow to the end of a path
        # p.M(20, -40).L(20, -27).L(0, -20)  # Chain multiple path operations
        # d.append(p)
        # d.append(draw.Line(30, -20, 0, -10,
        #             stroke='red', stroke_width=2, fill='none',
        #             marker_end=arrow))  # Add an arrow to the end of a line

        # d.setPixelScale(2)  # Set number of pixels per geometry unit
        # #d.setRenderSize(400,200)  # Alternative to setPixelScale
        # # d.saveSvg('example.svg')
        d.savePng(self.visual_file)

        # # Display in Jupyter notebook
        # d.rasterize()  # Display as PNG
        # d  # Display as SVG

    def show(self, ms=2000):
        def close_event():
            plt.close()

        f = plt.figure()
        timer = f.canvas.new_timer(interval=ms)
        timer.add_callback(close_event)

        img = mpimg.imread(self.visual_file)
        plt.imshow(img)

        timer.start()
        plt.show()


if __name__ == "__main__":
    task = GridWorldCarry([(0,0), (1,2), (5,6), (4,2)], (2,2), (0,3), (3,0), board_size=7)
    task.render_state(task.state_number)
    task.show()

    # REPL
    while True:
        print("Enter and action (0-9):")
        x = input()
        if x == "s":
            task.show()
        elif x == "q":
            break
        else:
            try:
                action = int(x)
            except:
                continue
            if not action in {0,1,2,3,4,5,6,7,8,9}:
                continue
            else:
                new_state_number = task.transition(task.state_number, action)
                assert(task.is_valid_state(new_state_number))
                task.set_state(new_state_number)
                

                task.render_state(task.state_number)
                reward = task.reward(task.state_number)
                if reward > 0.:
                    print(f"REWARD: {reward}")
                # task.show()


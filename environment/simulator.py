import numpy as np
import math

from environment.car import Car
from environment.map import get_map, map_amount
from shapely.geometry import LineString, Point, MultiPoint
from shapely.geometry.base import BaseGeometry
from game.action_space import id_to_action





class Simulator(object):
    """
    Runs a simulation of a game
    """

    def __init__(self, 
                 map_size, map_front_segments, map_back_segments,
                 checkpoint_reward, time_reward, lose_reward, win_reward,
                 lap_amount, checkpoint_max_time,
                 agent_size, agent_sensor_angles, agent_sensor_lengths):
        super().__init__()

        #Define random number generator
        self.rng = np.random.default_rng()
        
        #Define map size
        self.map_size = map_size
        
        #Define how many map segments to render in front/behind
        self.map_front_segments = map_front_segments
        self.map_back_segments = map_back_segments
        
        #Define rewards
        self.checkpoint_reward = checkpoint_reward
        self.time_reward = time_reward
        self.lose_reward = lose_reward
        self.win_reward = win_reward
        
        #Define maximum laps and step
        self.lap_amount = lap_amount
        self.checkpoint_max_time = checkpoint_max_time

        #Define agent size and sensors
        self.agent_size = agent_size
        self.agent_sensor_angles = agent_sensor_angles
        self.agent_sensor_lengths = agent_sensor_lengths


        #Initailize environment
        self.reset()


    def reset(self, map_id=None, map_direction=None):
        #If no map ID provided, get random map
        if map_id is None:
            self.map_id = self.rng.integers(0, map_amount)
        #Else, get requested map
        else:
            self.map_id = map_id
            
        #If no map direction provided, get random direction
        if map_direction is None:
            self.map_direction = self.rng.choice([True, False])
        #Else, get requested map
        else:
            self.map_direction = map_direction

        self.map = get_map(self.map_id, self.map_direction, *self.map_size)


        #Set segment and time agent is at
        self.current_segment = 0
        self.checkpoint_time_counter = 0

        #Calculate spawn point
        spawn_left = self.map[0, self.current_segment]
        spawn_right = self.map[1, self.current_segment]
        spawn_point = spawn_left + (spawn_right - spawn_left) / 2
        
        #Calculate spawn orientation
        next_left = self.map[0, self.current_segment + 1]
        next_right = self.map[1, self.current_segment + 1]
        next_point = next_left + (next_right - next_left) / 2

        spawn_direction = next_point - spawn_point
        spawn_orientation = np.arccos(spawn_direction[0] / np.linalg.norm(spawn_direction))
        #If pointing down, correct orientation
        #NOTE: Screen coordinates have inverted y-axis
        if spawn_direction[1] >= 0:
            spawn_orientation = 2*math.pi - spawn_orientation


        #Spawn agent
        self.car = Car(spawn_point, spawn_orientation, self.agent_size)


        #Return state
        return self.get_state()


    @property
    def window_size(self):
        return self.map_back_segments + self.map_front_segments
    
    @property
    def map_segments(self):
        return self.map.shape[1]
    
    @property
    def lap_counter(self):
        return self.current_segment // self.map_segments
    

    def get_window_walls(self):
        #NOTE: Assuming back/front segments are not long enough to wrap around the map fully.
        wall_begin = (self.current_segment - self.map_back_segments) % self.map_segments
        wall_end = (wall_begin + self.window_size + 1) % self.map_segments
        
        #If map has not ended, map is continous
        if wall_end > wall_begin:
            window_both = self.map[:, wall_begin:wall_end]
            window_left, window_right = window_both[0], window_both[1]
        #Else, wrap around map and append start to end
        else:
            wall_last = self.map[:, :wall_end]
            wall_first = self.map[:, wall_begin:]

            window_left = np.r_[wall_first[0], wall_last[0]]
            window_right = np.r_[wall_first[1], wall_last[1]]


        #Return window line. Reversing left wall to enable continous drawing
        return np.r_[window_left[::-1], window_right]


    def get_checkpoint(self):
        return self.map[:, (self.current_segment + 1) % self.map_segments]

    def get_agent_body(self):
        return self.car.get_rect()
    
    def get_sensors(self):
        rays = np.zeros((len(self.agent_sensor_angles), 2, 2))

        for i, angle in enumerate(self.agent_sensor_angles):
            #Get direction of ray offset from agent direction
            ray_direction = self.car.get_screen_direction(offset=angle)
            #Scale ray vector
            ray = ray_direction * self.agent_sensor_lengths[i]

            #Start point of ray is agent
            rays[i, 0] = self.car.position
            #End point is ray vector + agent
            rays[i, 1] = ray + self.car.position

        #Return all rays
        return rays


    def get_window_walls_line(self):
        return LineString(self.get_window_walls())
    
    def get_checkpoint_line(self):
        return LineString(self.get_checkpoint())
    
    def get_agent_body_line(self):
        return LineString(self.get_agent_body())
    
    def get_sensor_lines(self, input_sensors = None):
        #If sensor coordinates were provided, use them
        #NOTE: To reduce computation time
        if input_sensors is not None:
            sensors = input_sensors
        #Else, retrieve sensor coordinates
        else:
            sensors = self.get_sensors()

        #Turn coordinates into line strings
        lines = []
        for ray in sensors:
            lines.append(LineString(ray))

        #Return ray lines
        return lines


    def update_agent(self, time_delta, action):
        #Deconstruct into action
        action_tuple = id_to_action[action]

        #Update agent
        self.car.update(time_delta, action_tuple)


    def check_checkpoint(self, checkpoint_line, agent_body_line):
        return checkpoint_line.intersects(agent_body_line)
    
    def check_collision(self, window_line, agent_body_line):
        return window_line.intersects(agent_body_line)
    
    def check_time(self):
        return self.checkpoint_time_counter >= self.checkpoint_max_time
    
    def check_win(self):
        return self.lap_counter >= self.lap_amount
    
    def get_state(self, window_line = None):
        #If window line is not provided, get it
        #NOTE: Can be provided to avoid double computation
        if window_line is None:
            window_line = self.get_window_walls_line()
        
        
        #Get rays and ray lines
        rays = self.get_sensors()
        ray_lines = self.get_sensor_lines(rays)
        
        #Prepare state array
        state = np.zeros(len(ray_lines) + 2)
        # state = np.zeros(len(ray_lines))
        
        #Iterate through each ray
        for i, ray in enumerate(ray_lines):
            #Get intersection of ray with window
            cross = window_line.intersection(ray)


            #If there are multiple intersections, take closest
            if type(cross) is MultiPoint:
                distances = np.linalg.norm(np.array(cross) - self.car.position, axis=1)
                cross = cross[np.argmin(distances)]
                
            #If there is an interesection
            if type(cross) is Point:
                #Calculate how close intersection is to agent 
                # relative to length of ray
                direction = self.car.position - np.array([cross.x, cross.y])
                #Store relative length
                state[i] = np.linalg.norm(direction) / np.linalg.norm(rays[i, 1] - self.car.position)
            #Else, there is no intersection, set state for ray to max distance
            else:
                state[i] = 1.0

        #Store agent velocity angle relative to agent front
        state[-2] = self.car.get_drift_angle()
        #Store agent speed relative to agent size
        state[-1] = np.linalg.norm(self.car.velocity) / self.car.size[0]

        #Return final state
        return state


    def step(self, time_delta, action):
        #Initialize variables
        reward = 0
        done = False
        
        
        #Execute action and update agent
        self.update_agent(time_delta, action)


        #Get lines to check
        checkpoint_line = self.get_checkpoint_line()
        window_line = self.get_window_walls_line()
        agent_body_line = self.get_agent_body_line()


        #Get state of simulation
        state = self.get_state(window_line)
        
        #Update step
        self.checkpoint_time_counter += 1

        #Add time reward
        reward += self.time_reward

        #If checkpoint hit, add reward, move checkpoint and reset checkpoint time
        if self.check_checkpoint(checkpoint_line, agent_body_line):
            reward += self.checkpoint_reward
            self.current_segment += 1
            self.checkpoint_time_counter = 0
        
        #If lost, give reward and mark terminal
        if self.check_collision(window_line, agent_body_line) or self.check_time():
            reward += self.lose_reward
            done = True

        #NOTE: Can lose and win at the same time.
        #If won, give reward and mark terminal
        if self.check_win():
            reward += self.win_reward
            done = True


        #Return step information
        return state, reward, done

import numpy as np
import math

from environment.map import create_empty_map, create_middle_obstacle
from environment.agent import Agent
from environment.simulator import Simulator
from ai.spinbot import create_spinbot_controller


#Define distances
time_step = 16 #milliseconds
map_size = (1600, 900)
agent_start_edge_distance = 200
agent_size = np.array([50, 20])


##############################################################################################
#INSERT DEFINITION OF AI CONTROLLER
##############################################################################################
controllers = [
    create_spinbot_controller(),
    create_spinbot_controller()
]

map = create_empty_map(*map_size)
wall = create_middle_obstacle(*map_size)


#Helper function to create agents at specific positions and orientations
def create_agents():
    return [
        Agent(np.array([agent_start_edge_distance, map_size[1] / 2]),
            math.pi * 0,
            agent_size),
        Agent(np.array([map_size[0] - agent_start_edge_distance, map_size[1] / 2]),
            math.pi * 1,
            agent_size)
    ]

#Helper function to create simulation with required parameters
def create_simulation(agents):
    return Simulator(agents, controllers, [*map, wall])


#Create agents and simulation
agents = create_agents()
sim = create_simulation(agents)


#Game specific state
running = True
winner_found = False

#Game loop
while running:
    if winner_found:
        #Reset simulation
        agents = create_agents()
        sim = create_simulation(agents)

        winner_found = False


    #Simulate time step
    losers = sim.update(time_step)


    #If winner found
    if len(losers):
        winner_found = True
        #NOTE: Biased towards player 1 losing
        winner = 1 if losers[0] == 0 else 0

        ##############################################################################################
        #DO SOMETHING WHEN WINNER HAS BEEN FOUND
        ##############################################################################################

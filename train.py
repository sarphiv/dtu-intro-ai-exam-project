import numpy as np
import random as r

from environment.map import create_empty_map, create_middle_obstacle, create_spawns
from environment.agent import Agent
from environment.simulator import Simulator
from ai.predictive_seeker import create_predictive_seeker_controller


#Define distances
time_step_range = (6, 19) #milliseconds
map_size = (1600, 900)
agent_start_edge_distance = 200
agent_size = np.array([50, 20])
agent_spawns = create_spawns(*map_size, agent_start_edge_distance)


##############################################################################################
#INSERT DEFINITION OF AI CONTROLLER
##############################################################################################
controllers = [
    create_predictive_seeker_controller(enemy_ids=range(2, 4)),
    create_predictive_seeker_controller(enemy_ids=range(2, 4)),
    create_predictive_seeker_controller(enemy_ids=range(0, 2)),
    create_predictive_seeker_controller(enemy_ids=range(0, 2)),
]

map = create_empty_map(*map_size)
wall = create_middle_obstacle(*map_size)


#Helper function to create agents at specific positions and orientations
def create_agents():
    return [
        Agent(*agent_spawns[0],
               agent_size),
        Agent(*agent_spawns[1],
               agent_size),
        Agent(*agent_spawns[3],
               agent_size),
        Agent(*agent_spawns[4],
               agent_size),
    ]

#Helper function to create simulation with required parameters
def create_simulation(agents):
    return Simulator(agents, controllers, [*map, wall])


#Create agents and simulation
agents = create_agents()
sim = create_simulation(agents)

#NOTE: Uneven teams biases second team
team_size = len(agents) // 2


#Game specific state
running = True
winners = None

#Game loop
while running:
    if winners is not None:
        ##############################################################################################
        #DO SOMETHING WHEN WINNER HAS BEEN FOUND
        ##############################################################################################
        print(winners)

        #Reset simulation
        winners = None
        agents = create_agents()
        sim = create_simulation(agents)


    #Simulate time step
    losers = sim.update(r.randint(*time_step_range))


    #Get indexes of agents alive
    alive_indexes = np.array(list(sim.alive_agents.keys()))
    #If all alive agents are of the same team, mark winner
    if np.all(alive_indexes < team_size) or np.all(alive_indexes >= team_size):
        winners = alive_indexes

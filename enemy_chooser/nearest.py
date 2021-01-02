import numpy as np

from environment.simulator import Simulator


def create_nearest(enemy_ids):
    def chooser(simulator: Simulator, agent_id, time_delta):
        #NOTE: Does not account for bearing of agent.
        # Only looks at positional distance
        
        #Get alive agents
        alive_agents = simulator.alive_agents
        #Get agent position
        agent_pos = simulator.agents[agent_id].position
        
        #Calculate distance from agent to enemies
        distances = {id: np.linalg.norm(alive_agents[id].position - agent_pos) 
                     for id in enemy_ids 
                     if id in alive_agents }

        #Find nearest enemy
        nearest_id = None
        nearest_distance = None
        for id, distance in distances.items():
            if nearest_distance is None or nearest_distance > distance:
                nearest_id = id
                nearest_distance = distance

        #Return nearest enemy, or None if no enemy
        return nearest_id


    return chooser
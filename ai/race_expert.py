import numpy as np
import math
from game.action_space import action_to_id


def race_expert(targets, first_target_index, agent_retriever):
    
    
    if not type(targets) == np.ndarray:
        targets = np.array(targets)
    
    def nextTarget(target_index):
        return (target_index + 1) % len(targets)
    
    def getTarget(target_index, agen_position):
        direction = targets[target_index] - agen_position
        if (direction[0]**2 + direction[1]**2)**0.5 > 10:
            return target_index
        else:
            return nextTarget(target_index)
    
    target_index = first_target_index
    
    
    def controller():
        nonlocal target_index
        
        #Get agent being controlled
        agent = agent_retriever()
        agent_position = agent.position
        
        #Choose target
        target_index = getTarget(target_index, agent_position)
        target_position = targets[target_index]

        #Calculate direction and distance to target
        direction = target_position - agent_position
        distance = np.linalg.norm(direction)
        (x, y) = direction

        #Calculate bearing from agent to enemy
        #NOTE: Screen coordinates have inverted y-axis
        enemy_bearing = (math.acos(x / distance)) * (-1 if (y >= 0) else 1) % (2*math.pi)

        #Calculate bearing change needed
        bearing_change = (enemy_bearing - agent.bearing) % (2 * math.pi)
        bearing_change += -2*math.pi if bearing_change > math.pi else 0

        #Turn to face target
        left = bearing_change > 0
        right = not left

        #Forward if target in sight
        forward = abs(bearing_change) < math.pi / 128


        return action_to_id[(
            forward, #Forward
            False,   #Backward
            left,    #Left
            right,   #Right
        )]

    return controller
from environment.enemy_chooser import create_enemy_chooser
import numpy as np
import math
import random as r
from ai.idle import create_idle_controller

def create_seeker_controller(enemy_ids):
    #Initialize state variables for controller
    enemy_chooser = create_enemy_chooser(enemy_ids)
        
    #  Controller to activate if there are no enemies
    idle_controller = create_idle_controller()


    def controller(simulator, agent_id, time_delta):
        #Get agent being controlled
        agent = simulator.agents[agent_id]
        
        #Choose enemy
        enemy = enemy_chooser(simulator)
        #If there are no enemies, idle
        if enemy is None:
            return idle_controller(simulator, agent_id, time_delta)


        #Calculate direction and distance to target
        direction = enemy.position - agent.position
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

        #Shoot if target in sight
        shoot = abs(bearing_change) < math.pi / 128
        #Move forward while shooting
        forward = shoot


        return (
            forward, #Forward
            False,   #Backward
            left,    #Left
            right,   #Right
            shoot    #Shoot
        )

    return controller
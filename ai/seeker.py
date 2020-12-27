import numpy as np
import math

def create_seeker_controller():
    def controller(simulator, agent_id, time_delta):
        agent = simulator.agents[agent_id]
        #Mark first agent, which is not this agent, as the enemy
        enemy = [a for i, a in enumerate(simulator.agents) if i != agent_id][0]


        d = agent.position - enemy.position
        h = np.linalg.norm(d)
        (x, y) = d

        #Calculate bearing from agent to enemy
        #NOTE: Screen coordinates have inverted y-axis
        enemy_bearing = (math.acos(x / h)) * (-1 if (y >= 0) else 1) % (2*math.pi)

        #Calculate direction change needed
        direction_change = (enemy_bearing - agent.direction) % (2*math.pi)

        #Decide on actions
        left = direction_change > math.pi
        right = not left

        shoot = abs(direction_change) - math.pi < math.pi / 128
        forward = shoot


        return (
            forward, #Forward
            False,   #Backward
            left,    #Left
            right,   #Right
            shoot    #Shoot
        )

    return controller
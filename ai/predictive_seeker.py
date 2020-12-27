import numpy as np
import math

def create_predictive_seeker_controller():
    def controller(simulator, agent_id, time_delta):
        agent = simulator.agents[agent_id]
        #Mark first agent, which is not this agent, as the enemy
        enemy = [a for i, a in enumerate(simulator.agents) if i != agent_id][0]

        #Direction and distance to enemy
        direction = agent.position - enemy.position
        distance = np.linalg.norm(direction)

        #If gun available, set direction and distance to aim ahead of enemy's movement
        #NOTE: Using wrong distance to calculate travel time because trigonometry at night is too much.
        if not agent.cooldown_active:
            predict_direction = agent.position - (enemy.position + enemy.velocity * (distance / agent.bullet_speed))
            predict_distance = np.linalg.norm(predict_direction)
        #Else, move or turn towards enemy
        else:
            predict_direction = direction
            predict_distance = distance

        (x, y) = predict_direction


        #Calculate bearing from agent to enemy
        #NOTE: Screen coordinates have inverted y-axis
        enemy_bearing = (math.acos(x / predict_distance)) * (-1 if (y >= 0) else 1) % (2*math.pi)

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
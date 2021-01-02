from environment.agent import Agent
from environment.simulator import Simulator
from enemy_chooser.nearest import create_nearest
from ai.idle import create_idle_controller
import numpy as np
import math
import random as r
from shapely.geometry import LineString, Polygon


def create_predictive_seeker_controller(enemy_ids):
    #Initialize constants for controller
    fudge_factor_base = 1.0005

    flank_cooldown_limit = 0.60
    
    flank_angle = math.pi / 3
    flank_front_bias = 1
    flank_back_bias = 2
    flank_left_bias = 1
    flank_right_bias = 1
    flank_front_distance_scaler = 14
    flank_back_distance_scaler = 10

    collision_raw_velocity_scaler = 16
    collision_ray_middle_scaler = 4
    collision_ray_side_scaler = 2
    collision_ray_side_angle = math.pi / 4
    collision_ray_back_relative_scaler = 1 / 2

    shoot_accuracy_limit = math.pi / 128
    shoot_turn_accuracy_limit = math.pi / 256


    #Initialize state variables for controller
    enemy_chooser = create_nearest(enemy_ids)
    enemy_id = None

    #  Parameters for flanking enemy while gun is on cooldown
    #    Left or right biased flank
    flank_side = None
    #    Forwards or backwards biased flank
    flank_front = None
    #    Counter for how long a flank strategy has been active
    flank_counter = 0

    #  Controller to activate if there are no enemies
    idle_controller = create_idle_controller()
    

    def detect_target_obstruction(simulator: Simulator, agent, enemy):
        target_line_of_sight = LineString([tuple(agent.position), tuple(enemy.position)])

        return simulator.intersects_kill_zone(target_line_of_sight)


    def detect_collision_alert(simulator: Simulator, agent: Agent, time_delta):
        #Set up collision rays
        # Set up side ray rotation arrays
        ray_rotator = np.array([[np.cos(collision_ray_side_angle), -np.sin(collision_ray_side_angle)],
                                [np.sin(collision_ray_side_angle),  np.cos(collision_ray_side_angle)]])
        ray_rotator_inv = np.linalg.inv(ray_rotator)
        # Get normalized front ray
        ray_norm = agent.get_screen_direction()
        # Get scaler for rays, depending on velocity, agent size, and aggression
        ray_scaler = np.linalg.norm(agent.velocity) * time_delta / agent.size[0] * collision_raw_velocity_scaler + 1
        # Create front rays
        rays_front = np.array([ray_norm * collision_ray_middle_scaler, 
                               np.matmul(ray_rotator, ray_norm) * collision_ray_side_scaler, 
                               np.matmul(ray_rotator_inv, ray_norm) * collision_ray_side_scaler]) * ray_scaler
        # Create back rays
        rays_back = -rays_front * collision_ray_back_relative_scaler

        # Concatenate rays and get their absolute positions
        rays = np.vstack((rays_front, rays_back)) * agent.size[0] + agent.position
        
        # Create ray lines from agent
        ray_lines = []
        for ray in rays:
            ray_lines.append(LineString([tuple(ray), tuple(agent.position)]))


        #Detect collisions with rays
        collision_front_alert = False
        collision_back_alert = False
        # Loop through all ray lines
        for i, ray_line in enumerate(ray_lines):
            #If ray line intersects a kill zone, activate collision alert
            if simulator.intersects_kill_zone(ray_line):
                # If front ray collided, activate front alert
                if i < len(ray_lines) // 2:
                    collision_front_alert = True
                # Else, back ray collided, activate back alert
                else:
                    collision_back_alert = True

            #If both alerts activated, break, no point in further checking
            if collision_front_alert and collision_back_alert:
                break


        #Return alerts
        return (collision_front_alert, collision_back_alert)


    def aim_predictive_direction(agent, enemy):
        #Direction and distance to enemy
        direction = enemy.position - agent.position
        distance = np.linalg.norm(direction)

        #Calculate target to aim ahead of enemy's movement
        #NOTE: Using wrong distance and assumptions to calculate travel time because trigonometry at night is too much.
        # Using fudge factor to correct for inaccuracies in model.
        
        #Aim slightly ahead/behind depending on velocity, distance, and whether enemy is approaching
        fudge_factor = fudge_factor_base**(distance / agent.bullet_speed)
        approaching = 0 < (direction**2 - (direction + enemy.velocity)**2)
        fudge = fudge_factor**(approaching*2-1)

        #Calculate actual bullet speed
        bullet_speed = np.linalg.norm(agent.bullet_speed * agent.get_screen_direction() + agent.velocity)
        #Calculate time till impact
        impact_time_delta = distance / bullet_speed
        #Calculate enemy displacement till impact
        enemy_displacement = fudge*enemy.velocity * impact_time_delta
        
        #Calculate direction from agent to enemy
        return enemy.position + enemy_displacement - agent.position - agent.velocity * impact_time_delta


    def flank_direction(agent, enemy, time_delta):
        #Bring controller state into scope
        nonlocal flank_counter
        nonlocal flank_side
        nonlocal flank_front
        
        #If time to change flank strategy, change it
        if flank_counter <= 0:
            #Reset flank strategy timer
            flank_counter = enemy.cooldown_time // 2
            #Choose flank strategy
            #NOTE: Biased towards flanking from the rear sides
            flank_side = r.choice([1] * flank_left_bias + [-1] * flank_right_bias)
            flank_front = r.choice([1] * flank_front_bias + [-1] * flank_back_bias)

        #Control distance to enemy based on flank strategy
        flank_scaler = flank_front_distance_scaler if flank_front == 1 else flank_back_distance_scaler

        #Direction vector, relative to enemy, of where to flank
        enemy_flank_direction = enemy.get_screen_direction(offset=flank_angle * flank_side) * flank_front * enemy.size[1] * flank_scaler
        #Enemy movement in this time step
        #NOTE: Assuming no enemy action
        enemy_movement = enemy.velocity * time_delta

        #Calculate and return direction from agent to flank position
        return enemy_flank_direction + enemy.position + enemy_movement - agent.position


    def bearing_change(agent: Agent, direction):
        #Distance to position
        distance = np.linalg.norm(direction)
        (x, y) = direction
        
        #Calculate bearing from agent to position
        #NOTE: y-axis has inverted screen coordinates
        direction_bearing = (math.acos(x / distance)) * (-1 if (y >= 0) else 1) % (2*math.pi)

        #Calculate bearing change needed
        bearing_change = (direction_bearing - agent.bearing) % (2 * math.pi)
        bearing_change += -2*math.pi if bearing_change > math.pi else 0

        #Return change needed to face direction
        return bearing_change


    def controller(simulator: Simulator, agent_id, time_delta):
        #Bring controller state into scope
        nonlocal flank_counter
        nonlocal enemy_id
        
        #Update flank counter by decrementing if above 0
        flank_counter -= time_delta * (flank_counter > 0)


        #Get agent being controlled
        agent = simulator.agents[agent_id]


        #If there is no alive enemy targeted, attempt to choose one
        if enemy_id is None or enemy_id in simulator.dead_agents:
            enemy_id = enemy_chooser(simulator, agent_id, time_delta)

        #If there are no enemies, idle
        if enemy_id is None:
            return idle_controller(simulator, agent_id, time_delta)
        #Else, mark enemy
        else:
            enemy = simulator.agents[enemy_id]


        #Get status of gun
        gun_available_soon = agent.cooldown_counter / agent.cooldown_time < flank_cooldown_limit
        gun_available = not agent.cooldown_active or gun_available_soon


        #Check if target is obstructed
        target_blocked = detect_target_obstruction(simulator, agent, enemy)
        #Check collision alerts
        (collision_front_alert, collision_back_alert) = detect_collision_alert(simulator, agent, time_delta)


        #Get strategy
        steer_to_shoot = gun_available or gun_available_soon
        steer_to_position = target_blocked or collision_front_alert


        #Calculate how to hit target
        target_direction = aim_predictive_direction(agent, enemy)
        target_bearing_change = bearing_change(agent, target_direction)


        #If clear shoot, aim ahead of enemy's movement
        if steer_to_shoot and not steer_to_position:
            move_bearing_change = target_bearing_change
        #Else gun unavailable, or target obstructed, move to flank new enemy
        else:
            #Choose new enemy
            enemy_id = enemy_chooser(simulator, agent_id, time_delta)
            enemy = simulator.agents[enemy_id]
            
            #Get direction to new enemy
            move_direction = flank_direction(agent, enemy, time_delta)
            move_bearing_change = bearing_change(agent, move_direction)


        #Shoot if target within sights
        shoot = not target_blocked and (abs(target_bearing_change) < shoot_accuracy_limit)

        #If on target, do not turn
        if shoot and abs(target_bearing_change) < shoot_turn_accuracy_limit:
            (left, right) = (False, False)
        #Else, turn to face target/move direction
        else:
            left = move_bearing_change > 0
            right = not left

        #Move forwards if shooting, flanking, or avoiding a back collision. Unless there is a collision in front
        #NOTE: Still moves forwards while steering to shoot, as it steers to shoot before gun is available.
        # This helps it avoid shots
        forward = (shoot or agent.cooldown_active or target_blocked or collision_back_alert) and not collision_front_alert

        #Move backwards if there is a collision in front
        #NOTE: There could be a collision both sides. Too bad.
        backward = collision_front_alert and not collision_back_alert


        #Return actions
        return (
            forward,  #Forward
            backward, #Backward
            left,     #Left
            right,    #Right
            shoot     #Shoot
        )


    return controller
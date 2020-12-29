from environment.enemy_chooser import create_enemy_chooser
from ai.idle import create_idle_controller
import numpy as np
import math
import random as r
from shapely.geometry import LineString, Polygon


def create_predictive_seeker_controller(enemy_ids):
    #Initialize constants for controller
    fudge_factor_base = 1.0004

    flank_cooldown_limit = 0.60
    
    flank_angle = math.pi / 3
    flank_front_bias = 1
    flank_back_bias = 2
    flank_left_bias = 1
    flank_right_bias = 1
    flank_front_distance_scaler = 14
    flank_back_distance_scaler = 10

    aggression = 24

    collision_ray_middle_scaler = 3
    collision_ray_side_scaler = 1
    collision_ray_side_angle = math.pi / 4
    collision_ray_back_relative_scaler = 1 / 2

    shoot_accuracy_limit = (math.pi / 128)


    #Initialize state variables for controller
    enemy_chooser = create_enemy_chooser(enemy_ids)

    #  Parameters for flanking enemy while gun is on cooldown
    #    Left or right biased flank
    flank_side = None
    #    Forwards or backwards biased flank
    flank_front = None
    #    Counter for how long a flank strategy has been active
    flank_counter = 0

    #  Controller to activate if there are no enemies
    idle_controller = create_idle_controller()
    

    def detect_target_obstruction(simulator, agent, enemy, time_delta):
        target_line_of_sight = LineString([tuple(agent.position), tuple(enemy.position)])
        for box in simulator.kill_zones:
            box_shape = Polygon(box)
            
            if target_line_of_sight.intersects(box_shape):
                return True

        return False


    def detect_collision_alert(simulator, agent, time_delta):
                #Set up collision rays
        # Set up side ray rotation arrays
        ray_rotator = np.array([[np.cos(collision_ray_side_angle), -np.sin(collision_ray_side_angle)],
                                [np.sin(collision_ray_side_angle),  np.cos(collision_ray_side_angle)]])
        ray_rotator_inv = np.linalg.inv(ray_rotator)
        # Get normalized front ray
        ray_norm = agent.get_screen_direction()
        # Get scaler for rays, depending on velocity, agent size, and aggression
        ray_scaler = np.linalg.norm(agent.velocity) * time_delta / agent.size[0] * r.randrange(1, aggression) + 1
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
        # Loop through all kill zones
        for box in simulator.kill_zones:
            box_shape = Polygon(box)
            
            # Loop through all ray lines
            for i, ray_line in enumerate(ray_lines):
                #If kill zone collides with ray line, activate collision alert
                if ray_line.intersects(box_shape):
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


    def controller(simulator, agent_id, time_delta):
        #Bring controller state into scope
        nonlocal flank_counter
        
        #Update flank counter by decrementing if above 0
        flank_counter -= time_delta * (flank_counter > 0)


        #Get agent being controlled
        agent = simulator.agents[agent_id]

        #Choose enemy
        enemy = enemy_chooser(simulator)
        #If there are no enemies, idle
        if enemy is None: 
            return idle_controller(simulator, agent_id, time_delta)


        #Get status of gun
        gun_available_soon = agent.cooldown_counter / agent.cooldown_time < flank_cooldown_limit
        gun_available = not agent.cooldown_active or gun_available_soon

        steer_to_shoot = gun_available or gun_available_soon


        #Check if target is obstructed
        target_blocked = detect_target_obstruction(simulator, agent, enemy, time_delta)
        #Check collision alerts
        (collision_front_alert, collision_back_alert) = detect_collision_alert(simulator, agent, time_delta)

        steer_to_position = target_blocked or collision_front_alert


        #If clear shoot, aim ahead of enemy's movement
        if steer_to_shoot and not steer_to_position:
            target_direction = aim_predictive_direction(agent, enemy)
        #Else gun unavailable, or target obstructed, move to flank enemy
        else:
            target_direction = flank_direction(agent, enemy, time_delta)


        #Calculate distance to target
        target_distance = np.linalg.norm(target_direction)
        (x, y) = target_direction

        #Calculate bearing from agent to target
        #NOTE: y-axis has inverted screen coordinates
        enemy_bearing = (math.acos(x / target_distance)) * (-1 if (y >= 0) else 1) % (2*math.pi)

        #Calculate bearing change needed
        bearing_change = (enemy_bearing - agent.bearing) % (2 * math.pi)
        bearing_change += -2*math.pi if bearing_change > math.pi else 0


        #Turn to face target
        left = bearing_change > 0
        right = not left

        #Shoot if target within sights
        shoot = not target_blocked and (abs(bearing_change) < shoot_accuracy_limit)
        #Move forwards if shooting, flanking, or avoiding a back collision. Unless there is a collision in front
        forward = (shoot or agent.cooldown_active or target_blocked or collision_back_alert) and not collision_front_alert
        #Move backwards if there is a collision in front
        #NOTE: There could be a collision both sides. Too bad.
        backward = collision_front_alert


        #Return actions
        return (
            forward,  #Forward
            backward, #Backward
            left,     #Left
            right,    #Right
            shoot     #Shoot
        )


    return controller
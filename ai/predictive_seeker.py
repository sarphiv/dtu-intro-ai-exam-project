from ai.idle import create_idle_controller
import numpy as np
import math
import random as r
from shapely.geometry import LineString, Polygon


def create_predictive_seeker_controller(enemy_ids):
    #Initialize constants for controller
    fudge_factor_base = 1.0005

    flank_cooldown_limit = 0.40
    
    flank_front_bias = 1
    flank_back_bias = 2
    flank_left_bias = 1
    flank_right_bias = 1
    flank_front_distance_scaler = 12
    flank_back_distance_scaler = 8

    aggression = 24

    collision_ray_front_scaler = 4
    collision_ray_side_scaler = 2
    collision_ray_side_angle = math.pi / 6
    collision_ray_back_relative_scaler = 1 / 2

    shoot_accuracy_limit = (math.pi / 128)


    #Initialize state variables for controller
    #  If there are enemies, choose a random one
    if len(enemy_ids):
        enemy_id = r.choice(enemy_ids)
    #  Else, set no enemies
    else:
        enemy_id = None

    #  Parameters for flanking enemy while gun is on cooldown
    #    Left or right biased flank
    flank_side = None
    #    Forwards or backwards biased flank
    flank_front = None
    #    Counter for how long a flank strategy has been active
    flank_counter = 0

    #    Controller to activate if there are no enemies
    idle_controller = create_idle_controller()


    def controller(simulator, agent_id, time_delta):
        #Bring controller state into scope
        nonlocal enemy_id
        nonlocal flank_counter
        nonlocal flank_side
        nonlocal flank_front

        #Update flank counter by decrementing if above 0
        flank_counter -= time_delta * (flank_counter > 0)


        #Get agent being controlled
        agent = simulator.agents[agent_id]


        #Mark first alive enemy as target
        if enemy_id is None or enemy_id in simulator.dead_agents:
            enemy_id = None
            enemy = None
            for id in enemy_ids:
                if id not in simulator.dead_agents:
                    enemy_id = id
                    enemy = simulator.agents[id]
                    break
        else:
            enemy = simulator.agents[enemy_id]

        #If there are no enemies, idle
        if enemy_id is None:
            return idle_controller(simulator, agent_id, time_delta)


        #Direction and distance to enemy
        direction = enemy.position - agent.position
        distance = np.linalg.norm(direction)

        #If gun available, set bearing to aim ahead of enemy's movement
        #NOTE: Using wrong distance to calculate travel time because trigonometry at night is too much.
        # Using fudge factor to correct for inaccurate distance.
        if not agent.cooldown_active or (agent.cooldown_counter / agent.cooldown_time < flank_cooldown_limit):
            #Aim slightly ahead/behind depending on velocity, distance, and whether enemy is approaching
            fudge_factor = fudge_factor_base**(distance / agent.bullet_speed)
            approaching = 0 < (direction**2 - (direction + enemy.velocity)**2)
            fudge = fudge_factor**(approaching*2-1)

            #Calculate actual bullet speed
            bullet_speed = np.linalg.norm(agent.bullet_speed * agent.get_screen_direction() + agent.velocity)
            #Calculate enemy displacement till impact
            enemy_displacement = fudge*enemy.velocity * (distance / bullet_speed)
            #Calculate direction from agent to enemy
            target_direction = enemy.position + enemy_displacement - agent.position
        #Else, gun is on cooldown, move to flank enemy
        else:
            #If time to change flank strategy, change it
            if flank_counter <= enemy.cooldown_time // 2:
                #Reset flank strategy timer
                flank_counter = 0
                #Choose flank strategy
                #NOTE: Biased towards flanking from the rear sides
                flank_side = r.choice([1] * flank_left_bias + [-1] * flank_right_bias)
                flank_front = r.choice([1] * flank_front_bias + [-1] * flank_back_bias)

            #Control distance to enemy based on flank strategy
            flank_scaler = flank_front_distance_scaler if flank_front == 1 else flank_back_distance_scaler

            #Direction vector, relative to enemy, of where to flank
            enemy_flank_direction = enemy.get_screen_direction(offset=math.pi/3 * flank_side) * flank_front * enemy.size[1] * flank_scaler
            #Enemy movement in this time step
            #NOTE: Assuming no enemy action
            enemy_movement = enemy.velocity * time_delta
            #Calculate direction from agent to flank position
            target_direction = enemy_flank_direction + enemy.position + enemy_movement - agent.position


        #Calculate distance agent needs to travel
        predict_distance = np.linalg.norm(target_direction)
        (x, y) = target_direction

        #Calculate bearing from agent to enemy
        #NOTE: y-axis has inverted screen coordinates
        enemy_bearing = (math.acos(x / predict_distance)) * (-1 if (y >= 0) else 1) % (2*math.pi)

        #Calculate bearing change needed
        bearing_change = (enemy_bearing - agent.bearing) % (2 * math.pi)
        bearing_change += -2*math.pi if bearing_change > math.pi else 0

        #Turn to face target
        left = bearing_change > 0
        right = not left


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
        rays_front = np.array([ray_norm * collision_ray_front_scaler, 
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
            for i, nose_line in enumerate(ray_lines):
                #If kill zone collides with ray line, activate collision alert
                if nose_line.intersects(box_shape):
                    # If front ray collided, activate front alert
                    if i < len(ray_lines) // 2:
                        collision_front_alert = True
                    # Else, back ray collided, activate back alert
                    else:
                        collision_back_alert = True

            #If both alerts activated, break, no point in further checking
            if collision_front_alert and collision_back_alert:
                break

        #Shoot if target within sights
        shoot = abs(bearing_change) < shoot_accuracy_limit
        #Move forwards if shooting, flanking, or avoiding a back collision. Unless there is a collision in front
        forward = (shoot or agent.cooldown_active or collision_back_alert) and not collision_front_alert
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
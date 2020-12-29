from ai.idle import create_idle_controller
from ai.spinbot import create_spinbot_controller
from ai.seeker import create_seeker_controller
from ai.predictive_seeker import create_predictive_seeker_controller
import math
import pygame as pg
import numpy as np
from time import sleep

from environment.agent import Agent
from environment.simulator import Simulator
from environment.map import create_empty_map, create_middle_obstacle, create_spawns
from game.keyboard_controller import create_keyboard_controller, wasd_control_scheme, uhjk_control_scheme
from game.drawers import draw_bullet, draw_agent, draw_kill_zone


#Define distances
map_size = (1600, 900)
agent_start_edge_distance = 200
agent_size = np.array([50, 20])
agent_spawns = create_spawns(*map_size, agent_start_edge_distance)

#Initialize pygame
pg.init()
window = pg.display.set_mode(map_size)
clock = pg.time.Clock()
font = pg.font.SysFont(None, 48)

pg_events = None


#Define simulation parameters
agent_colors = [
    (0, 127, 255),
    (0, 127, 255),
    (255, 127, 0),
    (255, 127, 0),
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

#Helper function to create controllers for agents
def create_controllers():
    return [
        ##############################################################################################
        #Choose controllers
        # - Keyboard controlled agent
        # - Idle AI agent
        # - Spinbot AI agent
        # - Seeker AI agent
        # - Predictive seeker AI agent
        ##############################################################################################

        # create_keyboard_controller(lambda: pg_events, wasd_control_scheme),
        # create_keyboard_controller(lambda: pg_events, uhjk_control_scheme)
        # create_idle_controller(),
        # create_spinbot_controller(),
        # create_seeker_controller(),
        create_seeker_controller(enemy_ids=range(2, 4)),
        create_seeker_controller(enemy_ids=range(2, 4)),
        create_predictive_seeker_controller(enemy_ids=range(0, 2)),
        create_predictive_seeker_controller(enemy_ids=range(0, 2)),
    ]

#Helper function to create simulation with required parameters
def create_simulation(agents, controllers):
    return Simulator(agents, controllers, [*map, wall])


#Create agents, controllers, and simulation
agents = create_agents()
controllers = create_controllers()
sim = create_simulation(agents, controllers)

#NOTE: Uneven teams biases second team
team_size = len(agents) // 2


#Game specific state
running = True
winners = None

#Game loop
while running:
    if winners is not None:
        #Prepare end text
        if len(winners) > 0:
            player_text = f"Player{'s' if len(winners)>1 else ''}"
            end_text = font.render(f"{player_text} {' and '.join([str(w+1) for w in winners])} won!", 
                                   True, #Antialiasing
                                   agent_colors[winners[0]]) #Color
        else:
            end_text = font.render(f"Draw!", True, (128, 128, 128))

        #Draw end text horizontally centered at top of screen
        (w, h) = map_size
        window.blit(end_text, ((w-end_text.get_rect().width)//2, h//10))

        #Render to window
        pg.display.update()


        #Reset simulation
        winners = None
        #Create agents, controllers, and simulation
        agents = create_agents()
        controllers = create_controllers()
        sim = create_simulation(agents, controllers)


        #Pause game
        #NOTE: Disgusting, I know, performance is not important here
        sleep(3)

        #Reset game clock
        clock = pg.time.Clock()


    #Get time and events
    time_delta = clock.tick()
    pg_events = pg.event.get()

    #Handle events
    for e in pg_events:
        if e.type == pg.QUIT:
            running = False


    #Simulate time step
    sim.update(time_delta)


    #Get indexes of agents alive
    alive_indexes = np.array(list(sim.alive_agents.keys()))
    #If all alive agents are of the same team, mark winner
    if np.all(alive_indexes < team_size) or np.all(alive_indexes >= team_size):
        winners = alive_indexes
    #Else, continue drawing game
    else:
        #Draw background
        window.fill((255, 255, 255))

        #Draw kill zones
        for zone in sim.kill_zones:
            draw_kill_zone(window, zone)

        #Draw bullets
        for bullet in sim.bullets:
            draw_bullet(window, bullet)

        #Draw alive agents
        for agent_id, agent in sim.alive_agents.items():
            color = agent_colors[agent_id]
            draw_agent(window, agent, color)


    #Update window
    pg.display.update()

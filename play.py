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
from environment.map import create_empty_map, create_middle_obstacle
from game.keyboard_controller import create_keyboard_controller, wasd_control_scheme, uhjk_control_scheme
from game.drawers import draw_bullet, draw_agent, draw_kill_box


#Define distances
window_size = (1600, 900)
agent_start_edge_distance = 200
agent_size = np.array([50, 20])
agent_start = [
    #Top left corner
    (np.array([agent_start_edge_distance, agent_start_edge_distance]),
     math.pi / 4 * -1),
    #Bottom left corner
    (np.array([agent_start_edge_distance, window_size[1] - agent_start_edge_distance]),
     math.pi / 4 * 1),

    #Left side, middle of map
    (np.array([agent_start_edge_distance, window_size[1] / 2]),
     math.pi * 0),


    #Top right corner
    (np.array([window_size[0] - agent_start_edge_distance, agent_start_edge_distance]),
     math.pi / 4 * -3),
    #Bottom right corner
    (np.array([window_size[0] - agent_start_edge_distance, window_size[1] - agent_start_edge_distance]),
     math.pi / 4 * 3),
    
    #Right side, middle of map
    (np.array([window_size[0] - agent_start_edge_distance, window_size[1] / 2]),
     math.pi * 1)
]

#Initialize pygame
pg.init()
window = pg.display.set_mode(window_size)
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

controllers = [
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
    create_predictive_seeker_controller(enemy_ids=range(2, 4)),
    create_predictive_seeker_controller(enemy_ids=range(2, 4)),
    create_predictive_seeker_controller(enemy_ids=range(0, 2)),
    create_predictive_seeker_controller(enemy_ids=range(0, 2)),
]

map = create_empty_map(*window_size)
wall = create_middle_obstacle(*window_size)


#Helper function to create agents at specific positions and orientations
def create_agents():
    return [
        Agent(*agent_start[0],
            agent_size,
            armor=20),
        Agent(*agent_start[1],
            agent_size,
            armor=20),
        Agent(*agent_start[3],
            agent_size,
            armor=20),
        Agent(*agent_start[4],
            agent_size,
            armor=20),
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
winner = None

#Game loop
while running:
    if winner is not None:
        #Prepare win text
        win_text = font.render(f"Player {winner+1} won!", True, agent_colors[winner])

        #Draw win text horizontally centered at top of screen
        (w, h) = window_size
        window.blit(win_text, ((w-win_text.get_rect().width)//2, h//10))

        #Render to window
        pg.display.update()


        #Reset simulation
        agents = create_agents()
        sim = create_simulation(agents)

        winner = None

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
        winner = alive_indexes[0]
    #Else, continue drawing game
    else:
        #Draw background
        window.fill((255, 255, 255))

        #Draw kill boxes
        for box in sim.kill_zones:
            draw_kill_box(window, box)

        #Draw bullets
        for bullet in sim.bullets:
            draw_bullet(window, bullet)

        #Draw alive agents
        for agent_id, agent in sim.alive_agents.items():
            color = agent_colors[agent_id]
            draw_agent(window, agent, color)


    #Update window
    pg.display.update()

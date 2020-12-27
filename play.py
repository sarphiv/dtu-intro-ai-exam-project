from ai.spinbot import create_spinbot_controller
from ai.seeker import create_seeker_controller
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

#Initialize pygame
pg.init()
window = pg.display.set_mode(window_size)
clock = pg.time.Clock()
font = pg.font.SysFont(None, 48)

pg_events = None


#Define simulation parameters
agent_colors = [
    (0, 127, 255),
    (255, 127, 0)
]

controllers = [
    create_keyboard_controller(lambda: pg_events, wasd_control_scheme),

    ##############################################################################################
    #Choose opponent
    # 1. Keyboard controlled agent
    # 2. Seeker AI agent
    ##############################################################################################
    #create_keyboard_controller(lambda: pg_events, uhjk_control_scheme)
    create_seeker_controller()
]

map = create_empty_map(*window_size)
wall = create_middle_obstacle(*window_size)


#Helper function to create agents at specific positions and orientations
def create_agents():
    return [
        Agent(np.array([agent_start_edge_distance, window_size[1] / 2]),
            math.pi * 0,
            agent_size),
        Agent(np.array([window_size[0] - agent_start_edge_distance, window_size[1] / 2]),
            math.pi * 1,
            agent_size)
    ]

#Helper function to create simulation with required parameters
def create_simulation(agents):
    return Simulator(agents, controllers, [*map, wall])


#Create agents and simulation
agents = create_agents()
sim = create_simulation(agents)


#Game specific state
running = True
winner_found = False

#Game loop
while running:
    if winner_found:
        #Reset simulation
        agents = create_agents()
        sim = create_simulation(agents)
        
        winner_found = False

        #Pause game
        #NOTE: Disgusting, I know, performance is not important here
        sleep(3)

        #Reset the clock
        clock = pg.time.Clock()


    #Get time and events
    time_delta = clock.tick()
    pg_events = pg.event.get()

    #Handle events
    for e in pg_events:
        if e.type == pg.QUIT:
            running = False


    #Simulate time step
    losers = sim.update(time_delta)


    #Draw background
    window.fill((255, 255, 255))


    #Draw kill boxes
    for box in sim.kill_boxes:
        draw_kill_box(window, box)

    #Draw bullets
    for bullet in sim.bullets:
        draw_bullet(window, bullet)

    #Draw agents
    for agent, color in zip(agents, agent_colors):
        draw_agent(window, agent, color)



    #If winner detected, draw win text
    if len(losers):
        winner_found = True
        
        #NOTE: Biased towards player 1 losing
        winner = 1 if losers[0] == 0 else 0
        win_text = font.render(f"Player {winner+1} won!", True, agent_colors[winner])
        
        (w, h) = window_size
        #Draw win text horizontally centered
        window.blit(win_text, ((w-win_text.get_rect().width)//2, h//10))


    #Update window
    pg.display.update()

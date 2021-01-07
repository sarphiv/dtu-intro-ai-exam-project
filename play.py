import pygame as pg
import numpy as np
import random as r
import math
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch as T

from game.keyboard_controller import create_keyboard_controller, wasd_control_scheme, uhjk_control_scheme
from game.drawers import draw_game
from setup import create_simulator, create_agent, map_size, time_step, randomize_map




#Define parameters
running = True

pg.init()
window = pg.display.set_mode(tuple(map_size))
clock = pg.time.Clock()

game_counter = 0
time_delta = time_step
pg_events = None


#Helper function to tidy up things
def handle_events():
    global randomize_map
    global time_delta
    global pg_events
    global running
    
    #Get time and events
    time_delta = time_step if time_step else clock.tick()
    pg_events = pg.event.get()

    #Handle events
    for e in pg_events:
        if e.type == pg.QUIT:
            running = False
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_m:
                randomize_map = not randomize_map



sim, state = create_simulator()
agent = create_agent()

#Create temporary keyboard controller
# controller = create_keyboard_controller(lambda: pg_events, wasd_control_scheme)



#Game loop
while running:
    #Handle pygame events
    handle_events()

    #Get next action
    action = agent.action(state)

    #Simulate time step
    state, reward, done = sim.step(time_delta, action + 6)

    #Draw simulator environment
    draw_game(window, sim, state)

    #If agent won/lost, renew agent and reset rendered environment
    if done:
        #NOTE: If failed to load agent from file, tries again.
        # Can happen if file is being updated.
        while True:
            try:
                agent = create_agent()
                break
            except:
                print("Failed while loading agent")
                time.sleep(1.6)
                continue
        #Retrieve agent from file again in case of updates
        #Reset environment
        state = sim.reset() if randomize_map else sim.reset(sim.map_id, sim.map_direction)

import pygame as pg
import numpy as np
import random as r
import math
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch as T

from game.keyboard_controller import create_keyboard_controller, wasd_control_scheme, dvorak_wasd_control_scheme
from game.drawers import draw_game
from setup import create_simulator, create_agent, map_size, time_step, randomize_map




#Define parameters
running = True
restart = False

zoomed = True
zoom_in_scale = 6.0
zoom_out_scale = 1.0
zoom_scale_target = zoom_in_scale
zoom_scale = zoom_out_scale
zoom_pos_target = None #Defaults to car
zoom_pos = map_size / 2
zoom_scale_speed = 0.08
zoom_pos_speed = 0.07


pg.init()
window = pg.display.set_mode(tuple(map_size))
clock = pg.time.Clock()

game_counter = 0
time_delta = time_step
pg_events = None


sim, state = create_simulator()
agent = create_agent()

# controller = create_keyboard_controller(lambda: pg_events, wasd_control_scheme)
# controller = create_keyboard_controller(lambda: pg_events, dvorak_wasd_control_scheme)


#Helper functions to tidy up things
def handle_events():
    global randomize_map
    global time_delta
    global pg_events
    global running
    global restart
    global zoomed
    global zoom_scale
    global zoom_pos
    
    #Set zoom
    zoom_pos_target = sim.car.position if zoomed else map_size / 2
    zoom_scale_target = zoom_in_scale if zoomed else zoom_out_scale

    zoom_scale += (zoom_scale_target - zoom_scale) * zoom_scale_speed
    zoom_pos += (zoom_pos_target - zoom_pos) * zoom_pos_speed


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
            if e.key == pg.K_z:
                zoomed = not zoomed
            if e.key == pg.K_r:
                restart = True
                    

def zoomer(points):
    return ((points - zoom_pos) * zoom_scale + map_size / 2).astype(np.int)


#Game loop
while running:
    #Handle game events
    handle_events()
    

    #Get next action
    action = agent.action(state)
    # action = controller()

    #Simulate time step
    state, reward, done = sim.step(time_delta, action)

    #Draw simulator environment
    draw_game(window, sim, state, zoomer)

    #If agent won/lost, renew agent and reset rendered environment
    if done or restart:
        #Retrieve agent from file again in case of updates
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
        restart = False
        #Reset environment
        state = sim.reset() if randomize_map else sim.reset(sim.map_id, sim.map_direction)

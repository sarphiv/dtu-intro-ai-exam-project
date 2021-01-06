import pygame as pg
import numpy as np
import random as r
import math

from environment.simulator import Simulator
from game.keyboard_controller import create_keyboard_controller, wasd_control_scheme, uhjk_control_scheme
from game.drawers import draw_game


#Define parameters
running = True
render = True
randomize_map = True
time_step = None #Set to None, to use frame time

pg.init()
window = pg.display.set_mode((1600, 900))
clock = pg.time.Clock()

time_delta = 0
pg_events = None


#Helper function to tidy up things
def handle_events():
    global render
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
            if e.key == pg.K_r:
                render = not render
            if e.key == pg.K_m:
                randomize_map = not randomize_map



#Create temporary keyboard controller
controller = create_keyboard_controller(lambda: pg_events, wasd_control_scheme)


#Create environment
sim = Simulator(map_size=window.get_size(), 
                map_front_segments=4, 
                map_back_segments=1,
                checkpoint_reward=0, 
                time_reward=0,
                lose_reward=0,
                win_reward=0, 
                lap_amount=3, 
                time_amount=60000,
                agent_size=np.array([20, 10]), 
                agent_sensor_angles=[0, math.pi/3, -math.pi/3, math.pi/9, -math.pi/9],
                agent_sensor_lengths=[180, 180, 180, 180, 180])




#Game loop
while running:
    #Handle pygame events
    handle_events()



    #Get next action
    action = controller()

    #Simulate time step
    state, reward, done = sim.step(time_delta, action)


    #If agent won/lost, reset environment
    if done:
        print("Next map")

        if randomize_map:
            sim.reset()
        else:
            sim.reset(sim.map_id, sim.map_direction)



    #If rendering enabled, draw game
    draw_game(render, window, sim, state)


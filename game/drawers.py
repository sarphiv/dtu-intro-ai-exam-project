from math import trunc
import pygame as pg
import numpy as np


checkpoint_color = (128, 255, 192)
checkpoint_thickness = 2
window_wall_color = (128, 128, 255)
window_wall_thickness = 3
wall_color = (192, 192, 192)
wall_thickness = 2

sensor_color = (192, 255, 128)
sensor_thickness = 2

sensor_point_color = (255, 64, 0)
sensor_point_radius = 3

agent_color = (0, 192, 255)
agent_front_color = (128, 64, 128)


def draw_window_walls(surface, coordinates):
    pg.draw.lines(surface, 
                  window_wall_color, 
                  False, #Open line
                  coordinates,
                  window_wall_thickness)

def draw_walls(surface, coordinates):
    #Draw left wall
    pg.draw.lines(surface, 
                  wall_color, 
                  True, #Closed line
                  coordinates[0],
                  wall_thickness)

    #Draw right wall
    pg.draw.lines(surface, 
                  wall_color, 
                  True, #Closed line
                  coordinates[1],
                  wall_thickness)
    

def draw_checkpoint(surface, coordinates):
    pg.draw.lines(surface,
                  checkpoint_color, 
                  False, #Open line
                  coordinates, 
                  checkpoint_thickness)

def draw_sensors(surface, state, sensors, agent_position):
    points = (sensors[:, 1] - agent_position) * state[:len(sensors)].reshape(-1, 1) + agent_position
    
    #Draw sensor lines
    for point in points:
        pg.draw.line(surface, sensor_color, agent_position.astype(np.int), point.astype(np.int), sensor_thickness)
    
    #Draw sensor points
    for point in points:
        pg.draw.circle(surface, sensor_point_color, point.astype(np.int), sensor_point_radius)
        

def draw_agent(surface, agent):
    #Draw body
    agent_rect = agent.get_rect()
    pg.draw.polygon(surface, agent_color, agent_rect)

    #Draw front indicator
    front_direction = agent.get_screen_direction()
    front_position = agent.position + front_direction * agent.size[0] / 2
    front_radius = agent.size[1] / 3
    pg.draw.circle(surface, agent_front_color, tuple(front_position.astype(np.int)), round(front_radius))



def draw_game(surface, simulator, state):
    #Draw background
    surface.fill((255, 255, 255))

    #Draw walls and checkpoint
    draw_walls(surface, simulator.map)
    draw_checkpoint(surface, simulator.get_checkpoint())
    draw_window_walls(surface, simulator.get_window_walls())
    
    #Draw sensors and agent
    draw_sensors(surface, state, simulator.get_sensors(), simulator.agent.position)
    draw_agent(surface, simulator.agent)
    
    #Update window
    pg.display.update()
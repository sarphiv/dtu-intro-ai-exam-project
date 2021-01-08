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

car_color = (0, 192, 255)


def draw_window_walls(surface, coordinates, zoomer):
    pg.draw.lines(surface, 
                  window_wall_color, 
                  False, #Open line
                  zoomer(coordinates),
                  window_wall_thickness)

def draw_walls(surface, coordinates, zoomer):
    #Draw left wall
    pg.draw.lines(surface, 
                  wall_color, 
                  True, #Closed line
                  zoomer(coordinates[0]),
                  wall_thickness)

    #Draw right wall
    pg.draw.lines(surface, 
                  wall_color, 
                  True, #Closed line
                  zoomer(coordinates[1]),
                  wall_thickness)
    

def draw_checkpoint(surface, coordinates, zoomer):
    pg.draw.lines(surface,
                  checkpoint_color, 
                  False, #Open line
                  zoomer(coordinates), 
                  checkpoint_thickness)

def draw_sensors(surface, state, sensors, car_position, zoomer):
    points = (sensors[:, 1] - car_position) * state[:len(sensors)].reshape(-1, 1) + car_position
    
    #Draw sensor lines
    for point in points:
        pg.draw.line(surface, sensor_color, zoomer(car_position), zoomer(point), sensor_thickness)
    
    #Draw sensor points
    for point in points:
        pg.draw.circle(surface, sensor_point_color, zoomer(point), sensor_point_radius)
        

def draw_car(surface, car, zoomer):
    #Draw body
    car_rect = car.get_rect()
    pg.draw.polygon(surface, car_color, zoomer(car_rect))




def draw_game(surface, simulator, state, zoomer):
    #Draw background
    surface.fill((255, 255, 255))

    #Draw walls and checkpoint
    draw_walls(surface, simulator.map, zoomer)
    draw_checkpoint(surface, simulator.get_checkpoint(), zoomer)
    draw_window_walls(surface, simulator.get_window_walls(), zoomer)
    
    #Draw sensors and car
    draw_sensors(surface, state, simulator.get_sensors(), simulator.car.position, zoomer)
    draw_car(surface, simulator.car, zoomer)
    
    #Update window
    pg.display.update()
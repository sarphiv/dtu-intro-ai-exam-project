from math import trunc
import pygame as pg
import numpy as np


def draw_bullet(surface, bullet):
    tracer_factor = 16
    pos = tuple(bullet.position.astype(np.int))
    tracer_end = tuple((bullet.position - bullet.velocity * tracer_factor).astype(np.int))

    #Draw tracer
    pg.draw.aaline(surface, (255, 160, 160), pos, tracer_end)
    
    #Draw bullet
    pg.draw.circle(surface, 
                   (255, 0, 127), 
                   pos, 
                   bullet.width)


def draw_agent(surface, agent, color):
    #Draw body
    agent_rect = agent.get_rect()
    pg.draw.polygon(surface, color, agent_rect)


    #Draw gun indicator
    (cool_width_scale, cool_height_scale) = (0.8, 0.6)

    if agent.cooldown_active:
        cooldown_percent = agent.cooldown_counter / agent.cooldown_time
        cooldown_color = (255, 0, 0)
    else:
        cooldown_percent = agent.cooldown_counter / agent.cooldown_heat_max
        cooldown_color = (min(max(round(255 * cooldown_percent), 0), 255), 0, min(max(round(255 * cooldown_percent), 0), 255))
        
    cooldown_front_rect = agent.get_rect(cooldown_percent * cool_width_scale, cool_height_scale)
    cooldown_back_rect = agent.get_rect(cool_width_scale, cool_height_scale)
    
    pg.draw.polygon(surface, (0, 0, 0), cooldown_back_rect)
    pg.draw.polygon(surface, cooldown_color, cooldown_front_rect)


    #Draw gun
    gun_direction = agent.get_screen_direction()
    gun_position = agent.position + gun_direction * agent.size[0] / 2
    gun_radius = agent.bullet_width * 3 / 2
    gun_color = (255, 0, 0) if agent.cooldown_active else (255, 0, 255)
    
    pg.draw.circle(surface, gun_color, tuple(gun_position.astype(np.int)), round(gun_radius))


def draw_kill_box(surface, box):
    pg.draw.polygon(surface, (0, 0, 0), box)
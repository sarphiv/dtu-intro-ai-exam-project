import math
import pygame
import numpy as np
from environment.agent import Agent
from environment.simulator import Simulator


running = True

forward_down = False
backward_down = False
left_down = False
right_down = False
space_down = False

def keyboard_controller(simulator, time_delta):
    global running
    
    global forward_down
    global backward_down
    global left_down
    global right_down
    global space_down

    
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_w:
                forward_down = True
            elif e.key == pygame.K_s:
                backward_down = True
            elif e.key == pygame.K_a:
                left_down = True
            elif e.key == pygame.K_d:
                right_down = True
            elif e.key == pygame.K_SPACE:
                space_down = True
        elif e.type == pygame.KEYUP:
            if e.key == pygame.K_w:
                forward_down = False
            elif e.key == pygame.K_s:
                backward_down = False
            elif e.key == pygame.K_a:
                left_down = False
            elif e.key == pygame.K_d:
                right_down = False
            elif e.key == pygame.K_SPACE:
                space_down = False

    
    if forward_down:
        agent.move(True, time_delta)
    elif backward_down:
        agent.move(False, time_delta)
    
    if left_down:
        agent.turn(True, time_delta)
    elif right_down:
        agent.turn(False, time_delta)
        
    if space_down:
        bullet = simulator.agents[0].shoot()
        if bullet is not None:
            simulator.bullets.append(bullet)

window_size = (1600, 900)

pygame.init()
window = pygame.display.set_mode(window_size)

clock = pygame.time.Clock()

agent = Agent(np.array([400, 400]),
              0,
              np.array([50, 30]))

(w, h) = window_size
edge_depth = 100
edge_right = [(0, 0), (-edge_depth, 0), (-edge_depth, h), (0, h)]
edge_left = [(w, 0), (w+edge_depth, 0), (w+edge_depth, h), (w, h)]
edge_top = [(0, 0), (0, -edge_depth), (w, -edge_depth), (w, 0)]
edge_bottom = [(0, h), (0, h+edge_depth), (w, h+edge_depth), (w, h)]

sim = Simulator([agent], [keyboard_controller], [edge_right, edge_left, edge_top, edge_bottom])

while running:
    time_delta = clock.tick()

    window.fill((255, 255, 255))
    
    losers = sim.update(time_delta)
    
    if len(losers):
        print("lost")
    

    agent_rect = agent.get_rect()
    pygame.draw.polygon(window, (0, 0, 0), agent_rect)
    pygame.draw.circle(window, (255, 0, 0), (round(agent_rect[0][0]), round(agent_rect[0][1])), 5)
    d = agent.get_screen_direction()
    pos = agent.position + d * (agent.size[0] / 2 + agent.bullet_width * 3 / 2)
    pygame.draw.circle(window, (255, 0, 255), (round(pos[0]), round(pos[1])), agent.bullet_width)


    for bullet in sim.bullets:
        pygame.draw.circle(window, (255, 0, 0), tuple(bullet.position.astype(np.int)), bullet.width)

    
    pygame.display.update()

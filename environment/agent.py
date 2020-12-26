import math
import numpy as np

from environment.bullet import Bullet

class Agent(object):
    """
    Agent with ability to fight and shoot
    """
    
    def __init__(self, 
                 position, 
                 direction, 
                 size,
                 move_speed=0.002,
                 turn_speed = 0.005,
                 velocity = np.zeros(2),
                 smoothness = 0.997,
                 recoil = 0.05,
                 burst_cooldown = 70,
                 cooldown_heat = 200,
                 cooldown_heat_max = 2000,
                 cooldown_time = 1200,
                 instability = 1, 
                 bullet_speed = 3,
                 bullet_width = 3,
                 bullet_damage = 2.6):
        super().__init__()

        self.position = position
        self.direction = direction
        self.size = size
        
        self.move_speed = move_speed
        self.turn_speed = turn_speed
        self.velocity = velocity
        self.smoothness = smoothness
        self.recoil = recoil
        self.burst_cooldown = burst_cooldown
        self.burst_counter = 0
        self.cooldown_heat = cooldown_heat
        self.cooldown_heat_max = cooldown_heat_max
        self.cooldown_time = cooldown_time
        self.cooldown_active = False
        self.cooldown_counter = 0
        self.instability = instability
        
        self.bullet_speed = bullet_speed
        self.bullet_width = bullet_width
        self.bullet_damage = bullet_damage


    def update(self, time_delta):
        s = self.smoothness
        
        #p = p + v*s + v*s*s + v*s*s*s ... v*s**td
        #p = p + v*(s + s*s + s*s*s ... s**td)
        #p = p + v*((s^(td+1) - s) / (s - 1))
        self.position = self.position + self.velocity * ((s**(time_delta+1)-s)/(s-1))
        self.velocity = self.velocity * s**time_delta
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= time_delta
            if self.cooldown_active and self.cooldown_counter <= 0:
                self.cooldown_active = False

        if self.burst_counter > 0:
            self.burst_counter -= time_delta


    def get_screen_direction(self):
        # Negate y-axis because y-axis is reversed
        return np.array([ np.cos(self.direction), 
                         -np.sin(self.direction)])
    
    def get_screen_rotation(self):
        return np.array([[ np.cos(self.direction), np.sin(self.direction)],
                         [-np.sin(self.direction), np.cos(self.direction)]])
    
    def get_rect(self):
        (w, h) = self.size

        r = self.get_screen_rotation()
        relative_corners = np.matmul(r, np.array([[-w, -w, w,  w], 
                                                  [-h,  h, h, -h]]) / 2)

        absolute_corners = relative_corners + np.tile(self.position, (4, 1)).T

        return [(x, y) for x, y in zip(*absolute_corners)]


    def turn(self, counter_clockwise, time_delta):
        if counter_clockwise:
            self.direction += self.turn_speed * time_delta
        else:
            self.direction -= self.turn_speed * time_delta
            
        self.direction = self.direction % (math.pi * 2)


    def move(self, forward, time_delta):
        #Create direction vector
        d = self.get_screen_direction()
        
        #If moving backwards, reverse direction
        if not forward:
            d = -d

        #Update velocity
        self.velocity = self.velocity + d * self.move_speed * time_delta


    def shoot(self):
        #If gun on cooldown, do not shoot
        if self.cooldown_active:
            return None
        #Else if gun on burst cooldown, do not shoot
        elif self.burst_counter > 0:
            return None
        #Else handle gun cooldown
        else:
            self.cooldown_counter += self.cooldown_heat
            if self.cooldown_counter > self.cooldown_heat_max:
                self.cooldown_active = True
                self.cooldown_counter = self.cooldown_time

            self.burst_counter = self.burst_cooldown
        
        #Create direction vector
        d = self.get_screen_direction()

        #Recoil on agent
        self.velocity -= d * self.recoil

        #Spawn bullet at front of agent half a radius from tip
        b_pos = self.position + d * (self.size[0] / 2 + self.bullet_width * 3 / 2)
        #Fire bullet forwards and inherit agent velocity
        b_vel = d * self.bullet_speed + self.velocity

        #Return new bullet
        return Bullet(b_pos, b_vel, self.bullet_width, self.bullet_damage)
    
    def impact(self, bullet):
        self.instability += bullet.damage
        self.velocity += bullet.velocity * self.instability


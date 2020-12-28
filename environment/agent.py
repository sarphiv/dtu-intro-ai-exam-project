import math
import numpy as np
import copy

from environment.bullet import Bullet


class Agent(object):
    """
    Agent with ability to fight and shoot
    """
    
    def __init__(self, 
                 position, 
                 bearing, 
                 size,
                 move_acceleration=0.0024,
                 turn_speed = 0.004,
                 velocity = np.zeros(2),
                 velocity_resistance = 0.003,
                 instability = 0,
                 armor = 1,
                 recoil = 0.05,
                 burst_cooldown_time = 70,
                 burst_cooldown_counter = 0,
                 cooldown_heat = 200,
                 cooldown_heat_max = 2000,
                 cooldown_time = 1200,
                 cooldown_counter = 0,
                 bullet_speed = 2,
                 bullet_width = 2,
                 bullet_damage = 0.1):
        super().__init__()

        self.position = position
        self.bearing = bearing
        self.size = size

        self.move_acceleration = move_acceleration
        self.turn_speed = turn_speed
        self.velocity = velocity.copy()
        self.velocity_resistance = velocity_resistance

        self.instability = instability
        self.armor = armor
        self.recoil = recoil

        self.burst_cooldown_time = burst_cooldown_time
        self.burst_cooldown_counter = burst_cooldown_counter
        self.cooldown_heat = cooldown_heat
        self.cooldown_heat_max = cooldown_heat_max
        self.cooldown_time = cooldown_time
        self.cooldown_active = False
        self.cooldown_counter = cooldown_counter
        
        self.bullet_speed = bullet_speed
        self.bullet_width = bullet_width
        self.bullet_damage = bullet_damage


    def deep_copy(self):
        return copy.deepcopy(self)


    def update(self, time_delta):
        #Shorthand variable for resistance of movement
        t = time_delta
        r = self.velocity_resistance
        v = self.velocity
        p = self.position

        #Expression for velocity
        # dv/dt = -v*r
        
        # dv/dt + v*r = 0
        # v = c*e**(-r*t)
        
        # v(0) = c_v = v_s
        
        # v = v_s*e**(-r*t)

        #Expression for position
        # dp/dt = v

        # p = v/-r + c = v_s*e**(-r*t) / -r + c

        # p(0) = v_s/-r + c_p
        # p(0) + v_s/r = c_p = p_s + v_s/r

        # p = v_s*e**(-r*t) / -r + c_p

        c_v = v
        c_p = c_v/r + p

        self.velocity = c_v*math.exp(-r*t)
        self.position = c_v*math.exp(-r*t) / -r + c_p


        #Reduce cooldown counters
        if self.cooldown_counter > 0:
            self.cooldown_counter -= time_delta
            if self.cooldown_active and self.cooldown_counter <= 0:
                self.cooldown_active = False

        if self.burst_cooldown_counter > 0:
            self.burst_cooldown_counter -= time_delta


    def get_screen_direction(self, offset=0.0):
        #Get unit direction vector relative to screen coordinates
        #NOTE: Negate y-axis because y-axis is reversed
        return np.array([ np.cos(self.bearing + offset), 
                         -np.sin(self.bearing + offset)])
    
    def get_screen_rotation(self):
        #Get ortonormal matrix to rotate relative to screen coordinates
        return np.array([[ np.cos(self.bearing), np.sin(self.bearing)],
                         [-np.sin(self.bearing), np.cos(self.bearing)]])
    
    def get_rect(self, scale_width = 1, scale_height = 1):
        """
        Get bounding box for agent
        """
        
        #Get width and height of agent scaled by some factors (default: x=1, y=1)
        (w, h) = np.matmul(np.diag([scale_width, scale_height]), self.size)

        #Rotate points of bounding box to match with screen rotation
        r = self.get_screen_rotation()
        relative_corners = np.matmul(r, np.array([[-w, -w, w,  w], 
                                                  [-h,  h, h, -h]]) / 2)

        #Get absolute coordinates for bounding box corners
        absolute_corners = relative_corners + np.tile(self.position, (4, 1)).T

        #Return coordinates in a list of (x,y)-tuples
        return [(x, y) for x, y in zip(*absolute_corners)]


    def turn(self, counter_clockwise, time_delta):
        #If turning counter-clockwise, change direction with turn speed
        if counter_clockwise:
            self.bearing += self.turn_speed * time_delta
        #Else, turn clockwise
        else:
            self.bearing -= self.turn_speed * time_delta
        
        #Normalize direction to be [0, 2pi[
        self.bearing = self.bearing % (math.pi * 2)


    def move(self, forward, time_delta):
        #Create direction vector
        d = self.get_screen_direction()
        #If moving backwards, reverse direction
        if not forward:
            d = -d
        #Calculate acceleration vector
        a = d * self.move_acceleration

        #Update velocity
        self.velocity += a * time_delta


    def shoot(self):
        #If gun on cooldown, do not shoot
        if self.cooldown_active:
            return None
        #Else if gun on burst cooldown, do not shoot
        elif self.burst_cooldown_counter > 0:
            return None
        #Else handle gun cooldown
        else:
            self.cooldown_counter += self.cooldown_heat
            if self.cooldown_counter > self.cooldown_heat_max:
                self.cooldown_active = True
                self.cooldown_counter = self.cooldown_time

            self.burst_cooldown_counter = self.burst_cooldown_time
        
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
        #Receive damage
        self.instability += bullet.damage / self.armor
        #Update velocity with bullet velocity multiplied by health
        self.velocity += bullet.velocity * self.instability


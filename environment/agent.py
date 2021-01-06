import math
import numpy as np
import copy


class Agent(object):
    """
    Agent with ability to move
    """
    
    def __init__(self, 
                 position, 
                 bearing, 
                 size,
                 move_forward_acceleration=0.0009,
                 move_backward_acceleration=0.0003,
                 turn_speed = 0.002,
                 velocity = np.zeros(2),
                 velocity_resistance = 0.002,
                 drift_speed_limit = 1e-1):
        super().__init__()

        self.position = position
        self.bearing = bearing
        self.size = size

        self.move_forward_acceleration = move_forward_acceleration
        self.move_backward_acceleration = move_backward_acceleration
        self.turn_speed = turn_speed
        self.velocity = velocity.copy()
        self.velocity_resistance = velocity_resistance
        self.drift_speed_limit = drift_speed_limit


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

        self.velocity = c_v*np.exp(-r*t)
        self.position = c_v*np.exp(-r*t) / -r + c_p


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
        return np.array([[x, y] for x, y in zip(*absolute_corners)])
    
    
    def get_drift_angle(self):
        #Calculate speed
        speed = np.linalg.norm(self.velocity)

        #If not moving, return zero because not drifting
        if speed < self.drift_speed_limit:
            return 0

        #Deconstruct velocity
        (x, y) = self.velocity
        
        #Calculate bearing from agent to position
        #NOTE: y-axis has inverted screen coordinates
        velocity_bearing = (math.acos(x / speed)) * (-1 if (y >= 0) else 1) % (2*math.pi)

        #Calculate drift angle with sign needed
        drift_angle = (velocity_bearing - self.bearing) % (2 * math.pi)
        drift_angle += -2*math.pi if drift_angle > math.pi else 0


        #Return angle between velocity and agent front
        return drift_angle


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
        
        #Calculate acceleration vector
        if forward:
            a = d * self.move_forward_acceleration
        else:
            a = -d * self.move_backward_acceleration

        #Update velocity
        self.velocity += a * time_delta


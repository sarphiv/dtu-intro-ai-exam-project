import math
import numpy as np
import copy


class Car(object):
    """
    Car with ability to move
    """

    def __init__(self, 
                 position, 
                 bearing,
                 size,
                 velocity = np.zeros(2),
                 c_ = 0.000049,
                 c_v = 0.012,
                 c_vv = 0,
                 mass = 10,
                 engine_force_forward = 0.0005,
                 engine_force_backward = 0.0001,
                 turn_speed = 0.0005,
                 drift_speed_detection = 8e-3):
        super().__init__()

        self.position = position
        self.velocity = velocity.copy()
        self.bearing = bearing
        self.size = size

        self.c_ = c_
        self.c_v = c_v
        self.c_vv = c_vv
        self.mass = mass
        self.engine_force_forward = engine_force_forward
        self.engine_force_backward = engine_force_backward
        self.turn_speed = turn_speed
        self.drift_speed_detection = drift_speed_detection


    def deep_copy(self):
        return copy.deepcopy(self)


    def update(self, dt, action):
        
        (forward, backward, left, right) = action
        
        #turning
        if left:
            self.bearing += self.turn_speed % (math.pi * 2) * dt
        if right:
            self.bearing -= self.turn_speed % (math.pi * 2) * dt
        
        #moving
        speed = np.linalg.norm(self.velocity)
        u = self.get_screen_direction() # car unit vector

        F_t = u*self.engine_force_forward*forward
        F_b = -u*self.engine_force_backward*backward
        F_vv = -self.c_vv*speed*self.velocity
        F_v = -self.c_v*self.velocity
        F_  = -self.c_*self.velocity/speed if speed > 0 else 0
        
        F_sum = F_t + F_b + F_vv + F_v + F_

        a = F_sum/self.mass
        self.velocity += a*dt
        self.position += self.velocity*dt


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
        if speed < self.drift_speed_detection:
            return 0

        #Deconstruct velocity
        (x, y) = self.velocity
        
        #Calculate bearing from agent to position
        #NOTE: y-axis has inverted screen coordinates
        velocity_bearing = (math.acos(x / speed)) * (-1 if (y >= 0) else 1) % (2*math.pi)

        #Calculate drift angle
        drift_angle = (velocity_bearing - self.bearing) % (2 * math.pi)
        drift_angle += -2*math.pi if drift_angle > math.pi else 0


        #Return angle (without sign) between velocity and agent front
        return abs(drift_angle)

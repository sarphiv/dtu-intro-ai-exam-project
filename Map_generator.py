import matplotlib.pyplot as plt
import numpy as np
import random
import os

sign   = lambda n: abs(n)/n
retvek = lambda vinkel: np.array([np.cos(vinkel),np.sin(vinkel)])

def f(point,s1,s2,a1, u, p, sn):
    [x,y] = point
    sx = np.sin(x * s1) + np.sin(x * s2)
    sy = np.sin(y * s1) + np.sin(y * s2)
    a  = (y-np.sin(x*u-p))**2 + (x**2)/a1
    return sx + sy + a - sn


def track(check_length, bredde):
    #define variables
    s1 = random.uniform(0.3, 0.9)
    s2 = random.uniform(2.1, 3.1)
    a1 = random.uniform(4.0, 6.0)
    sn = random.uniform(7.0, 12.0)
    u  = random.uniform(0.7, 1.3)
    p  = random.uniform(0, 6.28)
    
    
    vinkel = 0
    
    #find first point
    point = np.array([0.,0.])
    step_size = 1
    for i in range(5):
        step = step_size/10**i  
        while True:
            point[1] += step
            if f(point,s1,s2,a1, u, p, sn) > 0:
                point[1] -= step
                break
    
    wall1 = []
    wall2 = []
    points = np.array([point])
    

    one_more = False
    
    while True:
        for i in range(4):
            step = step_size/10**i
            for j in range(10):
                
                next_point = points[-1] + check_length*retvek(vinkel)
                nextf = f(next_point,s1,s2,a1,u,p,sn)
                vinkel -= sign(nextf)*step
        
        #dot
        if len(wall1) > 2:
            t = points[-2:]-points[-3:-1]
            w1 = wall1[-2:]-wall1[-3:-1]
            w2 = wall2[-2:]-wall2[-3:-1]
            if np.dot(t[1],t[0]) < 0 or np.dot(w1[1], w1[0]) < 0 or np.dot(w2[1], w2[0]) < 0:
                return False, np.array([[1,0],[0,1]]), np.array([[0,0],[1,1]])
                return False, wall1, wall2
        #if np
        
            # print(next_point, vinkel, last_vinkel)
            # break
        
        points = np.vstack((points,next_point))
        midt = np.mean(points[-2:],axis = 0)
        vek = retvek(vinkel)[::-1]*np.array([-bredde,bredde])
        try:
            wall1 = np.vstack((wall1, midt + vek))
            wall2 = np.vstack((wall2, midt - vek))
        except:
            wall1 = midt + vek
            wall2 = midt - vek
        
        if one_more: break
        
        if len(points) > 6 and np.sum((points[-1] - points[0])**2)**0.5 <= 1.5*check_length: one_more = True
    
    points = np.array(points[:-1])
    wall1 = np.array(wall1)
    wall2 = np.array(wall2)
    
    return True, wall1, wall2

def plot(track_, i):
    fig, ax = plt.subplots()
    ax.plot(track_[0][:,0], track_[0][:,1])
    ax.plot(track_[1][:,0], track_[1][:,1])
    
    plt.title(str(i))
    
    ax.grid()
    
    #fig.savefig("test.png")
    plt.show()

def create_tracks(n, check_length, bredde):
    i = 0
    tracks = []
    while i < n:
        valid, wall1, wall2 = track(check_length, bredde)
        if valid: tracks.append([wall1,wall2]); i+=1
    return tracks
    
def save_tracks(tracks,filename):
    np.save(filename, tracks)




check_length = 0.6
bredde = 0.4

tracks = create_tracks(15, check_length, bredde)
for i, t in enumerate(tracks):
    plot(t,i); print(i)




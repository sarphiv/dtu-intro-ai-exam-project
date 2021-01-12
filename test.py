# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro')

# def init():
#     ax.set_xlim(0, 2*np.pi)
#     ax.set_ylim(-1, 1)
#     return ln,

# def update(frame):
#     xdata.append(frame)
#     ydata.append(np.sin(frame))
#     ln.set_data(xdata, ydata)
#     return ln,

# ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
#                     init_func=init, blit=True)
# plt.show()

# print(10)

# import matplotlib.pyplot as plt
# import numpy as np

# #x = np.linspace(0, 10*np.pi, 100)
# x = np.arange(10)
# y = lambda x : np.sin(x)

# print(np.shape(x), np.shape(y(x)))

# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1, = ax.plot(x, y(x), 'b-')
# print(line1)

# for phase in range(10, 200): #np.linspace(0, 10*np.pi, 100):
#     x = np.append(x, phase)
    
#     # print(phase)
#     #fig.clf()
#     #line1.set_xdata(x) #= ax.plot(x, np.sin(0.5 * x + phase), 'b-')
#     #line1.set_ydata(np.sin(0.5 * x))
#     line1.set_data(x, y(x))
    
#     # line1, = ax.plot(x, y(x), 'b-')
#     # print(line1)
#     #plt.figure().add_subplot(111).plot(x, y).
    
#     fig.canvas.draw()


# import datetime as dt
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # Create figure for plotting
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# xs = []
# ys = []

# count = lambda: dt.time

# # This function is called periodically from FuncAnimation
# def animate(i, xs, ys):

#     # Read temperature (Celsius) from TMP102
#     temp_c = count()

#     # Add x and y to lists
#     xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
#     ys.append(temp_c)

#     # Limit x and y lists to 20 items
#     xs = xs[-20:]
#     ys = ys[-20:]

#     # Draw x and y lists
#     ax.clear()
#     ax.plot(xs, ys)

#     # Format plot
#     plt.xticks(rotation=45, ha='right')
#     plt.subplots_adjust(bottom=0.30)
#     plt.title('TMP102 Temperature over Time')
#     plt.ylabel('Temperature (deg C)')

# # Set up plot to call animate() function periodically
# ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)
# plt.show()

import LunarLander 

ln = LunarLander.LunarLander()

from plotting.CsvManager import CsvManager
from plotting.PlotManager import PlotManager
from setup import plot_data_path, plot_x_axis, plot_y_axis
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time


#Load current plot file
plot_file = CsvManager([plot_x_axis, plot_y_axis], file_name=plot_data_path.format(0), clear=False)

#NOTE: Can add more plot files to plot them together
plots = PlotManager([plot_file])


#Show plot
while True:
    plots.plot()
    
    command = input("Press enter to update (q: quit)...")
    
    if command == 'q':
        break
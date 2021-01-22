import matplotlib.pyplot as plt
import numpy as np
from csv_manager import csv_manager

class plot_manager():
    def __init__(self, csv_managers):
        self.csv_managers = csv_managers
        self.colors = ["b", "r", "g", "c", "m", "y", "k"]
        
        # Editable plot  
        plt.ion()
        self.fig  = plt.figure()

        # plotting on one axis, could add plots above or below if wanted
        self.ax = self.fig.add_subplot(111)    
        
        
    def plot(self):
        plt.cla()
        
        # For each csv_manager plot their data 
        for i, manager in enumerate(self.csv_managers):
            data = manager.get_data()
            title = manager.get_header()
            
            setattr(self, f"line{i}", self.ax.plot(data[0], data[1], self.colors[i%7] + '-', label = title)[0])
        
        
        # Setting axis labels
        labels = self.csv_managers[0].get_titles() 
        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        #plt.legend(loc='upper left')
        #plt.legend(loc="lower right")
        
        # Show figure 
        plt.pause(0.00000001)
        self.fig.canvas.draw()   
        

# # test things : 
# csv_file = csv_manager(["data1", "data2"], clear=False)
# csv_file2 = csv_manager(["data1", "data4"], file_name="stats2.csv")
# plots = plot_manager([csv_file, csv_file2])
# for i in range(100):
#     #csv_file.save_data([i, i**2])
#     csv_file2.save_data([i, i**3 * 0.05])
#     plots.plot()
    

        
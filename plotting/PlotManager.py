import matplotlib.pyplot as plt
import numpy as np
from plotting.CsvManager import CsvManager

class PlotManager():
    def __init__(self, csv_managers, label=True):
        self.csv_managers = csv_managers
        self.label = label
        self.colors = ["b", "g", "m", "c", "r", "y", "k"]
        
        # Editable plot  
        plt.ion()
        self.fig  = plt.figure()

        # plotting on one axis, could add plots above or below if wanted
        self.ax = self.fig.add_subplot(111)    
        
        
    def plot(self, ):
        plt.cla()
        
        # For each csv_manager plot their data 
        for i, manager in enumerate(self.csv_managers):
            data = manager.get_data()
            title = manager.get_header()
            
            setattr(self, f"line{i}", self.ax.plot(data[0], data[1], self.colors[i%7] + '-', label = title,  alpha=0.54)[0])
        
        
        labels = self.csv_managers[0].get_titles() 
        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        
        # Setting axis labels
        if self.label:
            #plt.legend(loc='upper left')
            plt.legend(loc="lower right")
        
        # Show figure
        plt.pause(0.00000001)
        self.fig.canvas.draw()   
        
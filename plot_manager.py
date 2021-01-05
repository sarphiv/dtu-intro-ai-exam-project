import matplotlib.pyplot as plt
from matplotlib import animation
from csv_manager import csv_manager

class plot_manager():
    def __init__(self, csv_manager):
        self.csv_manager = csv_manager
        
        self.ani = animation.FuncAnimation(plt.gcf(), self.plot())
        
    def plot(self):
        data = self.csv_manager.get_data()
        
        plt.cla()
        plt.plot(data)
        # multiple lines / labels : 
        # plt.plot([data[0], data[1]], label=label1)
        # plt.plot([data[0], data[2]], label=label2)
        
        #plt.legend(loc='upper left')
        plt.tight_layout()
        

csv_file = csv_manager(["data1", "data2"])
plots = plot_manager(csv_file)

for i in range(100):
    csv_file.save_data([i, i**2])
    plots.plot()
    plt.show()
    

        
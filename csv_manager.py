import pandas as pd 
import csv

class csv_manager:
    
    def __init__(self, statistic_names, elements=2, file_name="statistic.csv"):
        # Initialising
        self.statistic_names = statistic_names
        self.elements = elements
        self.file_name = file_name
        
        # Writting headers 
        with open(file_name, 'w') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=statistic_names)
            csv_writer.writeheader()
        
        # Do i need to close it again? 
    
    
    def save_data(self, data): 
        # Write new datapoint 
        with open(self.file_name, 'w') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.statistic_names)
            csv_writer.writerow(dict(zip(self.statistic_names, data)))
    
    def get_data(self):
        # Get data in csv file as a numpy array 
        data = pd.read_csv(self.file_name)
        return data.to_numpy()    
     
        
    
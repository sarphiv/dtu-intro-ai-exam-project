import pandas as pd 
import csv

class CsvManager:
    
    def __init__(self, statistic_names, file_name="statistic.csv", clear=True):
        # Initialising
        self.statistic_names = statistic_names
        self.elements = len(statistic_names)
        self.file_name = file_name
        
        # Clear csv file if nothing else is specified 
        if clear:
            self.clear()
        
        # # Do i need to close it again? 
    
    def clear(self):
        # Rewritning the file 
        with open(self.file_name, 'w') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.statistic_names)
            csv_writer.writeheader()
    
    
    def save_data(self, data): 
        # Write new datapoint 
        with open(self.file_name, 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.statistic_names)
            csv_writer.writerow(dict(zip(self.statistic_names, data)))

    
    def get_data(self):
        # Get data in csv file as a numpy array 
        data = pd.read_csv(self.file_name).to_numpy().transpose()
        return data 
    
    def get_titles(self):
        # Get the column titles
        data = pd.read_csv(self.file_name)
        return list(data.columns) 
    
    def get_header(self):
        # Get the file_name 
        return self.file_name
     
        
    
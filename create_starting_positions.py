import pandas as pd
import csv
import numpy as np



validation_sets_names = ["random.csv", "choosen.csv", "outside.csv"]
validation_sets = []

for i in validation_sets_names: 
    validation_sets.append(pd.read_csv("starting_pos/" + i).to_numpy())


validation_counter = 1
(x, y, xspeed, yspeed, fuel) = set(validation_sets[validation_counter//30][validation_counter % 30])
print(x, y, xspeed, yspeed, fuel)

# file_name = "starting_pos/choosen.csv"
# headers = ["x", "y", "x_speed", "y_speed", "fuel"]

# with open(file_name, 'w') as csv_file:
#             csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
#             csv_writer.writeheader()
#             for i in range(30):
#                 x = 600*i/30 - 300 + 10
#                 y  = 400
#                 x_speed = 3*(np.random.rand()*2-1)
#                 y_speed = 3*(np.random.rand()*2-1)
#                 fuel = 100
#                 csv_writer.writerow(dict(zip(headers, [x,y,x_speed,y_speed,fuel])))
                

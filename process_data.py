import numpy as np 
import pandas as pd
import csv 
import os 
import math

path = "data/"

winning = []
avg_good_enough = []
always_good_enough = []

for i in range(1, 31):
    data = pd.read_csv(path + f"random_reward_{i}.csv").to_numpy().transpose()
    
    # Checking when the agent is winning on avg : 
    for j, c in enumerate(data[1]):
        if c >= 2000:
            winning.append(data[0, j])
            break
    # If not ever then -1 
    if len(winning) < i:
        winning.append(-1)
    
    # Checking when the agent becomes "good-enough" 
    for j, c in enumerate(data[1]):
        if c >= 4500:
            avg_good_enough.append(data[0, j])
            break
    # If not ever then -1 
    if len(avg_good_enough) < i:
        avg_good_enough.append(-1)
    
    data2 = pd.read_csv(path + f"random_won_{i}.csv").to_numpy().transpose()
    # Checking when the agent is "good-enough" 100 % of the games 
    for j, c in enumerate(data2[1]):
        if c >= 100:
            always_good_enough.append(data2[0, j])
            break
    # If never then -1 
    if len(always_good_enough) < i:
        always_good_enough.append(-1)

# print the lists : 
# print(winning)
# print(avg_good_enough)
# print(always_good_enough)
# input()

# converting to numpy for simplicity 
winning = np.array(winning)
avg_good_enough = np.array(avg_good_enough)
always_good_enough = np.array(always_good_enough)

# Writning down how many that didn't make it 
not_winning = np.count_nonzero(winning == -1)
winning = winning[winning != -1] 

not_avg_good_enough = np.count_nonzero(avg_good_enough == -1)
avg_good_enough = avg_good_enough[avg_good_enough != -1] 

not_always_good_enough = np.count_nonzero(always_good_enough == -1)
always_good_enough = always_good_enough[always_good_enough != -1] 

# Getting means, size and standard divations : 
winning_mean = winning.mean()
winning_size = len(winning)
winning_std = winning.std(ddof=-1)

avg_good_enough_mean = avg_good_enough.mean()
avg_good_enough_size = len(avg_good_enough)
avg_good_enough_std = avg_good_enough.std(ddof=-1)

always_good_enough_mean = always_good_enough.mean()
always_good_enough_size = len(always_good_enough)
always_good_enough_std = always_good_enough.std(ddof=-1)

# computing confidence interval : 
winning_error = 1.96*winning_std/np.sqrt(winning_size)
winning_lower = math.ceil(winning_mean - winning_error)
winning_upper = math.ceil(winning_mean + winning_error)

avg_good_enough_error = 1.96*avg_good_enough_std/np.sqrt(avg_good_enough_size)
avg_good_enough_lower = math.ceil(avg_good_enough_mean - avg_good_enough_error)
avg_good_enough_upper = math.ceil(avg_good_enough_mean + avg_good_enough_error)

always_good_enough_error = 1.96*always_good_enough_std/np.sqrt(always_good_enough_size)
always_good_enough_lower = math.ceil(always_good_enough_mean - always_good_enough_error)
always_good_enough_upper = math.ceil(always_good_enough_mean + always_good_enough_error)

print()
print(((winning-min(winning))/winning_mean).std(ddof=1))
# print(winning_mean, avg_good_enough_mean, always_good_enough_mean)
# print(winning_std**2, avg_good_enough_std**2, always_good_enough_std**2)
print(f"The confidence interval of winning with {not_winning} taken out: {winning_lower} - {winning_upper}")
print(f"Confidence from mean : {math.ceil(winning_mean)} +/- {math.ceil(winning_error)}")
print(f"The confidence interval of avg_good_enough with {not_avg_good_enough} taken out: {avg_good_enough_lower} - {avg_good_enough_upper}")
print(f"Confidence from mean : {math.ceil(avg_good_enough_mean)} +/- {math.ceil(avg_good_enough_error)}")
print(f"The confidence interval of always_good_enough with {not_always_good_enough} taken out: {always_good_enough_lower} - {always_good_enough_upper}")
print(f"Confidence from mean : {math.ceil(always_good_enough_mean)} +/- {math.ceil(always_good_enough_error)}")
print()


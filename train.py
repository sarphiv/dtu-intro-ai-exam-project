from plotting.CsvManager import CsvManager
import numpy as np
import random as r
import math
import os
from datetime import datetime
import torch as T
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from concurrent.futures import ProcessPoolExecutor
import copy

from environment.map import map_amount
from setup import create_policies, create_simulator, create_agent, time_step, randomize_map, policy_path, freeze_snapshot_folder, freeze_snapshot_file, epochs_per_freeze_snapshot, epoch_elite, epoch_training_iterations, epoch_training_data, epoch_offspring_per_elite, plot_data_path, plot_x_axis, plot_y_axis


#Simulation progress bar approximate length
simulation_bar_length = 64


def play_episodes_caller(params):
    return play_episodes(*params)

def play_episodes(policy, map_sequence, progress_blocks):
    #Keep track of progress bar state
    prev_progress_block = 0

    #Keep track of episode rewards
    episode_rewards = []


    #Create agent to play game
    agent = create_agent(policy)
    #Create simulator
    sim, state = create_simulator()
    #Create map sequence iterator
    map_data = iter(zip(*map_sequence))


    #Train agent x number of times based on y episodes
    for i in range(epoch_training_iterations):
        #Disable gradients for performancne
        T.no_grad()


        #Simulate episodes between training
        for e in range(epoch_training_data):
            #Saving episode trajectories
            states = []
            actions = []
            rewards = []

            done = False

            #Reset simulation and go to next map
            sim.reset(*next(map_data))


            #Play episode till completion
            while not done:
                #Get next action
                action = agent.action(state)

                #Simulate time step
                state, reward, done = sim.step(time_step, action)
                
                #Save rewards
                states.append(state)
                actions.append(action)
                rewards.append(reward)


            #Save episode
            agent.save_episode(states, actions, rewards)
            #Save episode reward
            episode_rewards.append(np.array(rewards).sum())

            #Update progress bar
            progress = (i*epoch_training_data + e + 1) / (epoch_training_iterations * epoch_training_data)
            if int(progress * progress_blocks) != prev_progress_block:
                print('=', end='', flush=True)
                prev_progress_block += 1


        #Reenable gradients for training
        T.enable_grad()

        #Train agent
        agent.train()


    #Return trained policy, and mean episode reward
    return policy, np.array(episode_rewards).mean()


def train():
    #Create simulation counters
    epoch_counter = 0
    last_save_point = 0
    
    #Create simulation data file
    plot_file = CsvManager([plot_x_axis, plot_y_axis], file_name=plot_data_path, clear=False)
    
    #Get random number generator
    rng = np.random.default_rng()

    #Store elite policies
    elites = create_policies()


    #Training loop
    while True:
        #Simulate episodes in parallel
        print("SIMULATING ", end='', flush=True)

        #Generate map sequence
        map_sequence_length = epoch_training_iterations*epoch_training_data
        #If map should be randomized, get randomized sequence
        if randomize_map:
            map_sequence = (rng.integers(0, map_amount, size=map_sequence_length), 
                            rng.choice([True, False], map_sequence_length))
        #Else, return constant map sequence
        else:
            map_sequence = (np.array([0] * map_sequence_length),
                            np.array([True] * map_sequence_length))


        #Calculate amount of progress bar blocks each simulation gets
        simulation_bar_blocks = simulation_bar_length / (epoch_elite * epoch_offspring_per_elite)


        #Make arguments for each simulation
        arguments = []
        for policy in elites:
            for _ in range(epoch_offspring_per_elite):
                arguments.append([copy.deepcopy(policy), map_sequence, simulation_bar_blocks])


        #Start simulations
        with ProcessPoolExecutor() as executor:
            simulation_batches = executor.map(play_episodes_caller, arguments)

        #Merge episode data
        print("> PROCESSING RESULTS", flush=True)
        simulation_batches = list(simulation_batches)

        #Sort simulations by elites first
        simulation_batches.sort(key=lambda b: b[1], reverse=True)

        #Store elites
        elites = [e for e, _ in simulation_batches[:epoch_elite]]


        #Update game counter
        epoch_counter += 1


        #Print and save best mean reward
        best_mean_reward = simulation_batches[0][1]

        print(f"{epoch_counter}: {best_mean_reward}", end='', flush=True)
        plot_file.save_data([epoch_counter, best_mean_reward])


        #Save snapshot of elites
        for i, policy in enumerate(elites):
            T.save(policy, policy_path.format(i))


        #If save checkpoint reached, store frozen snapshot of current elites
        if epoch_counter - last_save_point >= epochs_per_freeze_snapshot:
            #Reset save checkpoint counter
            last_save_point = epoch_counter
            
            #Prepare timestamp string
            timestamp = datetime.utcnow().isoformat().replace(':', '-')
            
            #Save snapshot of elites
            freeze_folder = freeze_snapshot_folder.format(timestamp, epoch_counter)
            os.mkdir(freeze_folder)
            for i, (policy, mean_reward) in enumerate(simulation_batches[:epoch_elite]):
                T.save(policy, freeze_folder + freeze_snapshot_file.format(i, mean_reward))


        #Print finish status message
        print(" ==> SAVED", end="\n\n", flush=True)



if __name__ == '__main__':
    train()

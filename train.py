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
from setup import create_policies, create_simulator, create_agent, time_step, randomize_map, policy_path, freeze_snapshot_folder, freeze_snapshot_file, epochs_per_freeze_snapshot, epoch_elite, epoch_training_iterations, epoch_training_data, epoch_evaluation, epoch_offspring_per_elite, plot_data_path, plot_x_axis, plot_y_axis



def create_map_sequence():
    #Get random number generator
    rng = np.random.default_rng()
    
    #Calculate length of map sequence based on training + evaluation
    map_sequence_length = epoch_training_iterations*epoch_training_data + epoch_evaluation

    #If map should be randomized, get randomized sequence
    if randomize_map:
        return (rng.integers(0, map_amount, size=map_sequence_length), 
                rng.choice([True, False], map_sequence_length))
    #Else, return constant map sequence
    else:
        return (np.array([0] * map_sequence_length),
                np.array([True] * map_sequence_length))


def simulate_map_caller(params):
    return simulate_map(*params)

def simulate_map(agent, map_id, map_direction):
    #Disable gradients for performancne
    T.no_grad()

    #Create simulator
    sim, state = create_simulator(map_id, map_direction)
    
    #Saving episode trajectories
    states = []
    actions = []
    rewards = []

    #Game completion state
    done = False

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

    #Return simulation results
    return states, actions, rewards


def simulate_maps(agent, map_sequence):
    #Create simulation arguments for each map
    arguments = [[agent, map_id, map_direction] for map_id, map_direction in map_sequence]

    #Start simulations
    with ProcessPoolExecutor() as executor:
        return executor.map(simulate_map_caller, arguments)


def train_agent_caller(params):
    return train_agent(*params)

def train_agent(policy, map_sequence):
    #Create agent to play game
    agent = create_agent(policy)

    #Create map sequence
    map_data = list(zip(*map_sequence))


    #Train agent for i training iterations with 'data' steps each time
    for i in range(0, epoch_training_iterations * epoch_training_data, epoch_training_data):
        #Disable gradients for performancne
        T.no_grad()

        #Simulate subset of map sequence with agent
        map_results = simulate_maps(agent, map_data[i:i + epoch_training_data])


        #Save results
        for states, actions, rewards in map_results:
            #Save episode
            agent.save_episode(states, actions, rewards)


        #Reenable gradients for training
        T.enable_grad()

        #Train agent
        agent.train()



    #Simulate evaluation subset of maps
    map_results = list(simulate_maps(agent, map_data[epoch_training_iterations * epoch_training_data:]))
    
    #Calculate mean episode summed reward
    episode_rewards = 0
    for _, _, rewards in map_results:
        episode_rewards += np.array(rewards).sum()

    episode_rewards = episode_rewards / len(map_results)


    #Print makeshift progress bar 
    print('=', end='', flush=True)

    #Return trained policy, and mean episode reward
    return policy, episode_rewards


def train():
    #Create simulation counters
    epoch_counter = 0
    last_save_point = 0
    
    #Create simulation data file
    plot_file = CsvManager([plot_x_axis, plot_y_axis], file_name=plot_data_path, clear=False)


    #Store elite policies
    elites = create_policies()


    #Training loop
    while True:
        #Simulate episodes in parallel
        print("SIMULATING ", end='', flush=True)

        #Generate map sequence
        map_sequence = create_map_sequence()

        #Make arguments for each simulation
        arguments = []
        for policy in elites:
            for _ in range(epoch_offspring_per_elite):
                arguments.append([copy.deepcopy(policy), map_sequence])

        #Start simulations
        with ProcessPoolExecutor() as executor:
            simulation_batches = executor.map(train_agent_caller, arguments)
            simulation_batches = list(simulation_batches)


        #Merge episode data
        print("> PROCESSING RESULTS", flush=True)

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

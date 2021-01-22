from plotting.CsvManager import CsvManager
import numpy as np
import random as r
import math
import os
from datetime import datetime
import torch as T
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from concurrent.futures import ProcessPoolExecutor
from game.action_space import index_to_action
import copy
from environment.LunarLanderSim import LunarLander

from setup import create_policies, create_agent, randomize_map, reward_factors, policy_path, freeze_snapshot_folder, freeze_snapshot_file, action_interval, epochs_per_freeze_snapshot, epoch_elite, epoch_training_iterations, epoch_training_data, epoch_evaluation, epoch_offspring_per_elite, max_generations, plot_data_path, plot_x_axis, plot_y_axis




NPROC_AGENTS = 12
NPROC_MAPS = 1


def create_start_sequence():
    #Get random number generator
    rng = np.random.default_rng()
    
    #Calculate length of map sequence based on training + evaluation
    map_sequence_length = epoch_training_iterations*epoch_training_data + epoch_evaluation

    #If map should be randomized, get randomized sequence
    if randomize_map:
        return [(300*(rng.uniform()*2-1), 3*(rng.uniform()*2-1), 3*(rng.uniform()*2-1)) for i in range(map_sequence_length)]
    #Else, return constant map sequence
    else:
        return [(-50, 2, -1) for _ in range(map_sequence_length)]


def simulate_map_caller(params):
    return simulate_map(*params)

def simulate_map(agent, start_state):
    #Disable gradients for performancne
    T.no_grad()

    #Create simulator
    env = LunarLander()
    env.reset(start_state)
    state = np.array(env.get_state())
    
    #Saving episode trajectories
    states = []
    actions = []
    rewards = []

    #Game completion state
    done = False
    
    i = 0
    
    #Play episode till completion
    while not done:
        take_action = i % action_interval == 0
        if take_action:
            #Get next action
            action_id = agent.action(state)
            action = index_to_action(action_id)

            #Save state and action
            states.append(state)
            actions.append(action_id)

        #Simulate time step
        (state, _, done) = env.step(action)
        (x, y, xspeed, yspeed, fuel) = state


        #If not done save rewards
        if not done and take_action:
            rewards.append(0)
            
        i += 1


    #Save terminal reward
    pos_delta = max(abs(x) - 20, 0) # within winning range 
    speed = (xspeed**2 + yspeed**2)**0.5 if not (xspeed**2 + yspeed**2)**0.5 - 20 < 0 else 0
    fuel = env.rocket.fuel
    won = env.won

    reward = np.array([pos_delta**0.25, speed, fuel*won, won])
    reward = (reward_factors * reward).sum()

    rewards.append(reward)


    #Return simulation results
    return states, actions, rewards


def simulate_maps(agent, start_sequence):
    global NPROC_MAPS
    #Create simulation arguments for each map
    arguments = [[agent, start_state] for start_state in start_sequence]

    #Start simulations
    return [simulate_map_caller(args) for args in arguments]
    # with ProcessPoolExecutor(NPROC_MAPS) as executor:
    #         return executor.map(simulate_map_caller, arguments)


def train_agent_caller(params):
    return train_agent(*params)

def train_agent(policy, start_sequence):
    #Create agent to play game
    agent = create_agent(policy)

    #Train agent for i training iterations with 'data' steps each time
    for i in range(0, epoch_training_iterations * epoch_training_data, epoch_training_data):
        #Disable gradients for performancne
        T.no_grad()

        #Simulate subset of map sequence with agent
        map_results = simulate_maps(agent, start_sequence[i:i + epoch_training_data])


        #Save results
        for states, actions, rewards in map_results:
            #Save episode
            agent.save_episode(states, actions, rewards)


        #Reenable gradients for training
        T.enable_grad()

        #Train agent
        agent.train()



    #Simulate evaluation subset of maps
    map_results = list(simulate_maps(agent, start_sequence[epoch_training_iterations * epoch_training_data:]))
    
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
    global NPROC_AGENTS

    #Create simulation counters
    epoch_counter = 0
    last_save_point = 0
    
    #Create simulation data file
    plot_file = CsvManager([plot_x_axis, plot_y_axis], file_name=plot_data_path, clear=False)


    #Store elite policies
    elites = create_policies()

    best_mean_reward = - 20000
    
    #Training loop
    while epoch_counter < max_generations:  #and best_mean_reward < 6000:
        #Simulate episodes in parallel
        print("SIMULATING ", end='', flush=True)

        #Generate map sequence
        start_sequence = create_start_sequence()

        #Make arguments for each simulation
        arguments = []
        for policy in elites:
            for _ in range(epoch_offspring_per_elite):
                arguments.append([copy.deepcopy(policy), start_sequence])

        #Start simulations
        simulation_batches = [train_agent_caller(args) for args in arguments]
        # with ProcessPoolExecutor(NPROC_AGENTS) as executor:
            # simulation_batches = executor.map(train_agent_caller, arguments)
            # simulation_batches = list(simulation_batches)


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

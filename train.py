from plotting.CsvManager import CsvManager
import numpy as np
import random as r
import math
import os
import shutil
from datetime import datetime
import torch as T
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from concurrent.futures import ProcessPoolExecutor

from setup import create_policy, create_simulator, create_agent, time_step, policy_folder, policy_path, freeze_snapshot_path, episodes_per_freeze_snapshot, num_agents, max_parallelism, max_episodes, plot_data_folder, plot_data_path, plot_x_axis, plot_y_axis



def train(id):
    episode_counter = 0
    last_save_point = 0

    #Create simulation data file
    plot_file = CsvManager([plot_x_axis, plot_y_axis], file_name=plot_data_path.format(id), clear=True)


    policy = create_policy(id)
    agent = create_agent(policy)


    #Create simulator
    sim, state = create_simulator()
    
    #Saving episode trajectories
    states = []
    actions = []
    rewards = []


    #Training loop
    while episode_counter < max_episodes:
        #Disable gradients for performancne
        T.no_grad()

        #Reset simulation
        state = sim.reset()

        states.clear()
        actions.clear()
        rewards.clear()
        
        #Game completion state
        done = False

        #Play episode till completion
        while not done:
            #Get next action
            action = agent.action(state)

            #Save state and action
            states.append(state)
            actions.append(action)

            #Simulate time step
            state, reward, done = sim.step(time_step, action)

            #Save rewards
            rewards.append(reward)


        #Save episode
        agent.save_episode(states, actions, rewards)

        #Reenable gradients for training
        T.enable_grad()

        #Train agent
        agent.train()


        #Update game counter
        episode_counter += 1

        #Calculate summed rewards for episode
        episode_reward = np.array(rewards).sum()

        #Save to plot file
        plot_file.save_data([episode_counter, episode_reward])


        #Save snapshot of policy
        T.save(policy, policy_path.format(id))


        #If save checkpoint reached, store frozen snapshot of agent
        if episode_counter - last_save_point >= episodes_per_freeze_snapshot:
            #Reset save checkpoint counter
            last_save_point = episode_counter
            
            #Prepare timestamp string
            timestamp = datetime.utcnow().isoformat().replace(':', '-')
            
            #Save snapshot of agent
            T.save(policy, freeze_snapshot_path.format(timestamp, episode_counter, id, episode_reward))


        #Print status
        print(f"{id}: {episode_counter}, {episode_reward}", flush=True)



    print(f"{id}: DONE=====================", flush=True)



if __name__ == '__main__':
    shutil.rmtree(policy_folder, ignore_errors=True)
    os.mkdir(policy_folder)

    shutil.rmtree(plot_data_folder, ignore_errors=True)
    os.mkdir(plot_data_folder)
    
    
    #Start simulations
    with ProcessPoolExecutor(max_parallelism) as executor:
        list(executor.map(train, [i for i in range(num_agents)]))

import numpy as np
import random as r
import math
import os
from datetime import datetime
import torch as T
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from concurrent.futures import ProcessPoolExecutor

from setup import create_simulator, create_agent, time_step, randomize_map, policy_path, freeze_snapshot_path



worker_episode_batch = 5
worker_procceses = 3
episodes_per_freeze_snapshot = 64
simulation_bar_length = 64


def start_worker(params):
    return play_episodes(*params)


def play_episodes(progress_blocks):
    #Disable gradients for performancne
    T.no_grad()

    #Create agent to play game
    agent = create_agent()

    #Keep track of progress bar state
    prev_progress_block = 0


    #Saving episode trajectories (list of lists)
    batch_states = []
    batch_actions = []
    batch_rewards = []

    #Create simulator
    sim, state = create_simulator()
    

    for i in range(worker_episode_batch):
        done = False

        #Reset episode memory
        episode_states = []
        episode_actions = []
        episode_rewards = []


        while not done:
            #Get next action
            action = agent.action(state)

            #Simulate time step
            state, reward, done = sim.step(time_step, action)
            
            #Save rewards
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)


        #Save episode
        batch_states.append(episode_states)
        batch_actions.append(episode_actions)
        batch_rewards.append(episode_rewards)


        #Reset simulator
        state = sim.reset() if randomize_map else sim.reset(sim.map_id, sim.map_direction)


        #Update progress bar
        if int((i+1) / worker_episode_batch * progress_blocks) != prev_progress_block:
            print('=', end='', flush=True)
            prev_progress_block += 1


    #Return episodes once done
    return batch_states, batch_actions, batch_rewards


def train():
    #Create agent to train
    agent = create_agent()
    
    #Create game counters
    game_counter = 0
    last_save_point = 0


    #Training loop
    while True:
        #Simulate episodes in parallel
        print("SIMULATING ", end='', flush=True)
        T.no_grad()
        with ProcessPoolExecutor() as executor:
            batches = executor.map(play_episodes, [simulation_bar_length // worker_procceses] * worker_procceses)


        #Merge episode data
        print("> COLLECTING BATCHES", flush=True)
        #Storage for summed reward
        summed_rewards = []
        
        #Iterate through each worker batch
        for batch in batches:
            #Get episodes from each batch
            batch_states, batch_actions, batch_rewards = batch
            
            #Iterate through each episode
            for states, actions, rewards in zip(batch_states, batch_actions, batch_rewards):
                agent.save_episode(states, actions, rewards)

                #Store mean reward
                summed_rewards.append(np.array(rewards).sum())


        #Update game counter
        game_counter += worker_procceses * worker_episode_batch

        #Print mean reward
        summed_mean = np.array(summed_rewards).mean()
        print(f"{game_counter}: {summed_mean}", end='')


        #Initiate training
        print(" ==> TRAINING", end='', flush=True)
        T.enable_grad()

        agent.train()


        #Save current agent
        T.save(agent.policy, policy_path)


        #If save checkpoint reached, store snapshot of current agent
        if game_counter - last_save_point >= episodes_per_freeze_snapshot:
            #Reset save checkpoint counter
            last_save_point = game_counter
            
            #Prepare timestamp string
            timestamp = datetime.utcnow().isoformat().replace(':', '-')

            #Save snapshot
            T.save(agent.policy, freeze_snapshot_path.format(timestamp, summed_mean))


        #Print finish status message
        print(" ==> SAVED", end="\n\n", flush=True)



if __name__ == '__main__':
    train()

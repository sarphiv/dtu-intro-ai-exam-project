import numpy as np
import random as r
import math
import os
import torch as T
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from concurrent.futures import ProcessPoolExecutor

from setup import create_simulator, create_agent, time_step, randomize_map, policy_path



worker_episode_batch = 20
worker_procceses = 4
training_repetitions = 2
simulation_bar_length = 5
training_bar_length = 6




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
            print('=', end='')
            prev_progress_block += 1


    #Return episodes once done
    return batch_states, batch_actions, batch_rewards


def train():
    #Create agent to train
    agent = create_agent()
    
    #Create game counter
    game_counter = 0

    while True:
        print("SIMULATING ", end='')
        T.no_grad()
        with ProcessPoolExecutor() as executor:
            batches = executor.map(play_episodes, [simulation_bar_length // worker_procceses] * worker_procceses)


        print("> COLLECTING BATCHES...")
        #Storage for mean reward
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
        print(f"{game_counter}: {np.array(summed_rewards).mean()}", end='')


        print(" ==> TRAINING ", end='')
        T.enable_grad()

        #Keep track of progress bar state
        prev_progress_block = 0

        for i in range(training_repetitions):
            agent.train()

            #Update progress bar
            if int((i+1) / training_repetitions * training_bar_length) != prev_progress_block:
                print('=', end='')
                prev_progress_block += 1

        T.save(agent.policy, policy_path)
        print("> SAVED", end="\n\n")



if __name__ == '__main__':
    train()

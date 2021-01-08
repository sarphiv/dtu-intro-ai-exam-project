import numpy as np
import torch as T
from ai.policy_gradient.Reinforce import Reinforce


class Agent(object):
    def __init__(self, 
                 policy: Reinforce, learning_rate,
                 future_discount, 
                 games_avg_store, games_avg_replay, 
                 replay_buffer_size, replay_batch_size):
        #Initialize super class
        super().__init__()
        
        #Create policy
        self.policy = policy
        
        #Create optimizer
        self.optimizer = T.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        #Create RNG
        self.rng = np.random.default_rng()
        
        #Set hyper-parameters
        self.future_discount = future_discount
        
        self.games_avg_store = games_avg_store
        self.games_avg_replay = games_avg_replay
        self.replay_buffer_size = replay_buffer_size
        self.replay_batch_size = replay_batch_size
        
        #Prepare memory
        self.state_memory = np.zeros((self.replay_buffer_size, self.policy.layer_dims[0]))
        self.action_memory = np.zeros((self.replay_buffer_size))
        self.expected_reward_memory = np.zeros((self.replay_buffer_size))
        
        self.memory_write_index = 0
        self.memory_filled = False
        
        self.games_steps_memory = np.zeros((0))


    def action(self, state):
        #Feed state forward through policy and get probabilities for actions
        probs = self.policy.forward(state)
        #Turn logits into a probability distribution
        action_probs = T.distributions.Categorical(probs=probs)
        #Sample one action from probability distribution
        action = action_probs.sample()


        #Return action
        return action.item()


    def __write_replay_buffer(self, states, actions, expected_rewards):
        #NOTE: The replay buffer is implemented as a ring buffer.
        # Hence the surprisingly complicated code below.
        
        #Get state of replay buffer
        mem_index = self.memory_write_index
        replay_size = self.replay_buffer_size
        #Limit write amount to size of replay buffer
        episode_length = min(len(states), replay_size)
        
        #Slice input to only contain most recent elements
        # that fit in the replay buffer
        states_slice = states[-episode_length:]
        actions_slice = actions[-episode_length:]
        expected_rewards_slice = expected_rewards[-episode_length:]
        
        #Calculate amount of elements to write 
        # before reaching end of ring buffer
        first_write_amount = min(replay_size - mem_index, episode_length)
        first_write_limit = mem_index + first_write_amount
        #Calculate amount of elements to write
        # after looping around in ring buffer
        last_write_amount = episode_length - first_write_amount
        
        #Execute first write from memory write index to end
        self.state_memory[mem_index:first_write_limit] = states_slice[:first_write_amount]
        self.action_memory[mem_index:first_write_limit] = actions_slice[:first_write_amount]
        self.expected_reward_memory[mem_index:first_write_limit] = expected_rewards_slice[:first_write_amount]
        
        #Execute second write from start to end of input data
        #NOTE: These lines only execute when the ring buffer loops around
        self.state_memory[0:last_write_amount] = states_slice[first_write_amount:]
        self.action_memory[0:last_write_amount] = actions_slice[first_write_amount:]
        self.expected_reward_memory[0:last_write_amount] = expected_rewards_slice[first_write_amount:]


        #Update memory filled and write index
        new_mem_index = (mem_index + episode_length) % replay_size
        if new_mem_index <= mem_index and episode_length > 0:
            self.memory_filled = True

        self.memory_write_index = new_mem_index


    def __read_replay_buffer(self, buffer, start, end):
        if start <= end:
            return buffer[start:end]
        else:
            first = buffer[start:]
            last = buffer[0:end]
            
            return np.concatenate((first, last))


    def save_episode(self, states, actions, rewards):
        #Update average game length estimate
        #NOTE: Inefficient, but average amount of games to store
        # should usually not be too big
        #Remove length of oldest game from front
        if len(self.games_steps_memory) >= self.games_avg_store:
            self.games_steps_memory = self.games_steps_memory[1:]

        #Add length of newest game to end
        self.games_steps_memory = np.append(self.games_steps_memory, len(states))
        #If episodes are approaching a size that could be too big for the replay buffer, print warning
        if self.games_steps_memory[-1] * self.games_avg_store > self.replay_buffer_size:
            print(f"WARN: Encountered episode with step amount ({self.games_steps_memory[-1]}) * amount of average games to store ({self.games_avg_store}) > replay buffer size ({self.replay_buffer_size})")


        #Initialize storage for expected rewards
        expected_rewards = np.zeros_like(rewards)
        #Set last expected reward as last reward
        expected_rewards[-1] = rewards[-1]

        #Calculate expected rewards backwards
        for i in reversed(range(len(rewards)-1)):
           expected_rewards[i] = rewards[i] + expected_rewards[i+1]*self.future_discount


        #Write to replay buffer
        self.__write_replay_buffer(np.array(states), 
                                   np.array(actions),
                                   expected_rewards)


    def train(self):
        #Calculate amount of steps to sample
        games_avg_steps = self.games_steps_memory.mean()
        steps_to_sample = int(self.games_avg_replay * games_avg_steps)
        #If too many steps for replay buffer, print warning
        if steps_to_sample > self.replay_buffer_size:
            print(f"WARN: Replay buffer maximum size {self.replay_buffer_size} too small for sampling {steps_to_sample}")


        #Get replay buffer sampling range
        sample_start_limit = self.memory_write_index - steps_to_sample
        sample_end_limit = self.memory_write_index
        
        #If replay buffer is filled and wrapping around
        if self.memory_filled and sample_start_limit < 0:
            #If more than a full wrap around is necessary, replay buffer too small
            if self.replay_buffer_size - abs(sample_start_limit) < sample_end_limit:
                print(f"WARN: Replay buffer {self.replay_buffer_size} is too small for sampling {steps_to_sample}")
                sample_start_limit = 0
                sample_end_limit = self.replay_buffer_size
        #Else, only use filled portion
        elif sample_start_limit < 0:
            print(f"WARN: Not enough unique samples {self.memory_write_index} in replay buffer for sampling {steps_to_sample}")
            sample_start_limit = 0


        #Randomly select recent samples from replay buffer
        sample_ids = self.rng.integers(sample_start_limit, sample_end_limit, size=steps_to_sample)
        sampled_expected_rewards = self.expected_reward_memory[sample_ids]
        mean = sampled_expected_rewards.mean()
        std = sampled_expected_rewards.std(ddof=1)
        
        #Backpropagate through samples in batches
        #NOTE: Necessary if e.g. GPU does not have enough memory
        for i in range(len(sample_ids) // self.replay_batch_size + 1):
            #Calculate batch indexes
            batch_start_index = i * self.replay_batch_size
            batch_end_index = batch_start_index + self.replay_batch_size
            
            #Get batch IDs
            batch_sample_ids = sample_ids[batch_start_index: batch_end_index]

            #If batch is empty, done processing
            if len(batch_sample_ids) == 0:
                break
    
            #Sample (state, action, expected_reward) triplets to train on
            states = T.tensor(self.state_memory[batch_sample_ids]).to(self.policy.device)
            actions = T.tensor(self.action_memory[batch_sample_ids]).to(self.policy.device)
            #NOTE: Standardizing rewards to reduce variance
            expected_rewards = self.expected_reward_memory[batch_sample_ids]
            standardized_rewards = (expected_rewards - mean) / (std if std else 1)
            standardized_rewards = T.tensor(standardized_rewards).to(self.policy.device)


            #Get action probabilities
            probs = self.policy.forward(states)
            action_log_probs = T.distributions.Categorical(probs=probs).log_prob(actions)


            #Calculate losses
            #NOTE: Gradient ascent on performance.
            # Same as gradient descent on negated performance
            negated_performance = -(action_log_probs * standardized_rewards).mean()

            self.optimizer.zero_grad()
            negated_performance.backward()
            self.optimizer.step()
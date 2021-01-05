import numpy as np
import torch as T
from PolicyGradient import PolicyGradient


class Agent(object):
    def __init__(self, 
                 policy: PolicyGradient, learning_rate,
                 future_discount, replay_buffer_size, replay_batch_size):
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
        self.replay_buffer_size = replay_buffer_size
        self.replay_batch_size = replay_batch_size
        
        #Prepare memory
        self.state_memory = np.zeros((self.replay_buffer_size, self.policy.layer_dims[0]))
        self.action_memory = np.zeros((self.replay_buffer_size))
        self.expected_reward_memory = np.zeros((self.replay_buffer_size))
        
        self.memory_write_index = 0
        self.memory_filled = False


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


    def save_episode(self, states, actions, rewards):
        #Calculate expected reward for each state
        expected_rewards = np.zeros_like(rewards)
        
        #Loop variables
        #TODO: O(N**2) - Refactor for better time complexity
        for i in range(len(rewards)):
            discount = 1
            expected_reward = 0

            for r in rewards[i:]:
                expected_reward += r * discount
                discount *= self.future_discount
                
            expected_rewards[i] = expected_reward


        self.__write_replay_buffer(np.array(states), 
                                   np.array(actions), 
                                   expected_rewards)


    def train(self):
        #If replay buffer is filled, use full replay buffer
        if self.memory_filled:
            episode_limit = self.replay_buffer_size
        #Else, only use filled portion
        else:
            episode_limit = self.memory_write_index

        #Sample (state, action, reward) triplets to train on
        episode_ids = self.rng.integers(0, episode_limit, size=self.replay_batch_size)
        
        states = T.tensor(self.state_memory[episode_ids]).to(self.policy.device)
        actions = T.tensor(self.action_memory[episode_ids]).to(self.policy.device)

        #Standardize rewards to reduce variance
        expected_rewards = self.expected_reward_memory[episode_ids]
        mean = expected_rewards.mean()
        std = expected_rewards.std(ddof=1)
        standardized_rewards = (expected_rewards - mean) / (std if std else 1)
        standardized_rewards = T.tensor(standardized_rewards).to(self.policy.device)

        #Get action probabilities
        probs = self.policy.forward(states)
        action_log_probs = T.distributions.Categorical(probs=probs).log_prob(actions)

        #Do gradient ascent on performance
        #NOTE: Same as gradient descent on negated performance
        negated_performance = -(action_log_probs * standardized_rewards).mean()

        self.optimizer.zero_grad()
        negated_performance.backward()
        self.optimizer.step()


import numpy as np
import torch
from typing import List


# batch_size = 64
learning_rate = 1e-5
future_discount = 0.99


def normalize_state(x, y, x_speed, y_speed, fuel):
    return [x / 800, y / 600, x_speed / 60, y_speed / 60, fuel / 100]

def standardize_rewards(rewards: np.ndarray):
    std = rewards.std(ddof=1)
    return (rewards - rewards.mean()) / (std if std else 1)


def calculate_expected_reward(rewards: List[int], terminals: List[int]) -> np.ndarray:
    padded_terminals = [-1] + terminals
    expected_rewards = []
    
    # Runs through every game and maps the expected reward 
    for prev_t, curr_t in enumerate(padded_terminals[1:]):
        trajectory_rewards = rewards[padded_terminals[prev_t]+1:curr_t+1]
        total_expected_reward = sum(trajectory_rewards)

        discount = future_discount
        for r in trajectory_rewards:
            expected_rewards.append(total_expected_reward * discount)
            discount *= future_discount
            total_expected_reward -= r

    #Return stantdardized expected rewards
    return standardize_rewards(np.array(expected_rewards))
    #return np.array(expected_rewards)
    

def train(model, states, actions, rewards, terminals):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = torch.nn.NLLLoss()


    t_states = torch.tensor(states)
    t_action = torch.tensor(actions)
    t_expected_rewards = torch.tensor(calculate_expected_reward(rewards, terminals))
    
    
    logits = model(t_states)
    #action_probs = logits / logits.sum(axis=1).reshape((-1, 1))
    action_probs = logits

    new_list = torch.zeros_like(action_probs, dtype=torch.bool)
    for i, c in enumerate(actions):
        new_list[i][c] = True
        
    # probs = action_probs[new_list].log()
    probs = action_probs[new_list].log()

    
    negated_reward = loss(logits.log(), torch.tensor([a[0] for a in actions], dtype=torch.long))
    #negated_reward = -(probs * t_expected_rewards).mean()
    
    optimizer.zero_grad()
    negated_reward.backward()
    print(negated_reward)
    optimizer.step()
    

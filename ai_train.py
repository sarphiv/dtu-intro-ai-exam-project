
import torch


# batch_size = 64
learning_rate = 1e-4


def train(model, states, actions, rewards):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    

    t_states = torch.tensor(states)
    t_action = torch.tensor(actions)
    
    #TODO: WRITE ME TO BE GENERAL FOR MANY TRAJECTORIES
    expected_reward = torch.tensor(rewards[0])
    
    
    logits = model(t_states)
    action_probs = logits / logits.sum(axis=1).reshape((-1, 1))

    new_list = torch.zeros_like(action_probs, dtype=torch.bool)
    for i, c in enumerate(actions):
        new_list[i][c] = True
        
    probs = action_probs[new_list].log()

    #TODO: Divide by number of games
    negated_reward = -(probs * expected_reward).sum() / 1
    
    optimizer.zero_grad()
    negated_reward.backward()
    print(negated_reward)
    optimizer.step()
    

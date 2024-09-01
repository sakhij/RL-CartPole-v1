import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 500
TARGET_UPDATE = 10

# Environment setup
env = gym.make('CartPole-v1')
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(args)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Select an action using epsilon-greedy policy
def select_action(state, policy_net, steps_done):
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * steps_done / EPSILON_DECAY)
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

# Optimize model using experience replay
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None  # No optimization if memory is not sufficient
    
    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))
    
    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])
    non_final_next_states = torch.cat([s for s in batch[3] if s is not None])
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[3])), dtype=torch.bool)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Return loss value for tracking

# Initialize policy and target networks
policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

# Training loop
num_episodes = 750
steps_done = 0
rewards_per_episode = []
loss_per_episode=[]

for episode in range(num_episodes):
    state = env.reset()[0]
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    total_reward = 0
    episode_loss = 0

    for t in range(200):
        action = select_action(state, policy_net, steps_done)
        next_state= env.step(action.item())[0]
        reward= env.step(action.item())[1]
        done= env.step(action.item())[2]
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32)
        
        total_reward += reward.item()
        rewards_per_episode.append(total_reward)
        
        if done:
            next_state = None
        
        memory.push(state, action, reward, next_state)
        state = next_state
        
        loss = optimize_model()
        if loss is not None:
            episode_loss += loss
            loss_per_episode.append(episode_loss)
        
        steps_done += 1
        
        if done:
            break
    
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Episode {episode} - Total Reward: {total_reward:.2f}, Loss: {episode_loss:.4f}")

env.close()

# Plot the rewards
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Performance on CartPole-v1')
plt.show()

plt.plot(loss_per_episode)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('DQN Loss on CartPole-v1')
plt.show()

# RL-CartPole-v1
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

# Q-Learning
 Q-Learning uses previously learned “states” which have been explored to consider future moves and stores this information in a “Q-Table.” For every action taken from a state, the policy table, Q table, has to include a positive or negative reward.
 
## Installations
Make sure that python3 and pip is installed. These libraries need to be installed.
```bash
pip install gymnasium numpy matplotlib
```

To verify the installations
```bash
pip freeze | grep -E 'gymnasium|numpy|matplotlib'
```

## How to Run
Clone the repository and run the python code:
```bash
https://github.com/sakhij/RL-CartPole-v1.git
```

## Expected Output:(Use as Reference)
Graphs for episodes where mean_rewards for last 100 episodes <1000.
![image](https://github.com/user-attachments/assets/2405ecd2-e337-4aff-90ae-1f79055b1380)
![image](https://github.com/user-attachments/assets/469eef01-fa41-41c9-a186-139e9a550998)
![image](https://github.com/user-attachments/assets/af8eb13f-f1e0-4144-845b-3454cf5cd9d8)

# Deep-Q Network
Train a DQN agent to balance a pole on a cart for as long as possible. This classic control problem serves as a benchmark to evaluate the performance of reinforcement learning algorithms.

## Installations
Make sure that python3 and pip is installed. These libraries need to be installed.
```bash
pip install torch numpy matplotlib
```
To verify the installations:
```bash
pip freeze | grep -E 'numpy|torch|matplotlib'
```
## How to Run:
Clone the repository and run the python code:
```bash
https://github.com/sakhij/RL-CartPole-v1.git
```

## Expected Output:(Use as Reference)
Graphs for 750 episodes.
![image](https://github.com/user-attachments/assets/a1022b55-cfcd-400a-9ad9-1fdc9ca93ec4)
![image](https://github.com/user-attachments/assets/f49e001a-3092-44d5-8f89-653a4ce851c4)


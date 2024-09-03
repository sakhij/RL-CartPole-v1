import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(is_training=True, render=False):

    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    if is_training:
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))  # init a 11x11x11x11x2 array
    else:
        with open('cartpole.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.1
    discount_factor_g = 0.99
    epsilon = 1
    epsilon_decay_rate = 0.00001
    rng = np.random.default_rng()

    rewards_per_episode = []
    losses_per_episode = []
    mean_rewards_per_episode = []
    std_rewards_per_episode = []

    i = 0

    while True:

        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False
        rewards = 0
        episode_loss = 0

        while not terminated and rewards < 10000:

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            # Calculate target and loss
            target = reward + discount_factor_g * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :])
            prediction = q[state_p, state_v, state_a, state_av, action]
            loss = (target - prediction) ** 2
            episode_loss += loss

            if is_training:
                q[state_p, state_v, state_a, state_av, action] = prediction + learning_rate_a * (target - prediction)

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward

        rewards_per_episode.append(rewards)
        losses_per_episode.append(episode_loss)

        mean_rewards = np.mean(rewards_per_episode[max(0, len(rewards_per_episode) - 100):])
        mean_rewards_per_episode.append(mean_rewards)
        std_rewards = np.std(rewards_per_episode[max(0, len(rewards_per_episode) - 100):])
        std_rewards_per_episode.append(std_rewards)

        if is_training and i % 100 == 0:
            print(f'Episode: {i}  Rewards: {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards: {mean_rewards:0.1f}')

        if mean_rewards > 1000:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        i += 1

    env.close()

    if is_training:
        with open('cartpole.pkl', 'wb') as f:
            pickle.dump(q, f)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    ax1.plot(rewards_per_episode, label='Reward per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode vs. Reward per Episode')
    ax1.legend()

    ax2.plot(losses_per_episode, label='Loss', color='r')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Episode vs. Loss')
    ax2.legend()
    
    ax3.plot(mean_rewards_per_episode, label='Mean Reward (Sliding Window)', color='g')
    ax3.fill_between(range(len(std_rewards_per_episode)), 
                     np.array(mean_rewards_per_episode) - np.array(std_rewards_per_episode),
                     np.array(mean_rewards_per_episode) + np.array(std_rewards_per_episode),
                     color='g', alpha=0.2, label='Stability Range')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Mean Reward')
    ax3.set_title('Learning Curve: Episode vs. Mean Reward')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('cartpole.png')
    plt.show()

if __name__ == '__main__':
    run(is_training=True, render=False)
    # run(is_training=False, render=True) #For testing the model

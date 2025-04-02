import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class GridWorld:
    def __init__(self, size=4, terminal_states=[(0, 0), (3, 3)], reward=-1):
        self.size = size
        self.terminal_states = terminal_states
        self.reward = reward
        self.state = (size-1, 0)
        
    def reset(self):
        self.state = (self.size-1, 0)
        return self.state_to_index(self.state)

    def state_to_index(self, state):
        return state[0] * self.size + state[1]

    def step(self, action):
        if self.state in self.terminal_states:
            return self.state_to_index(self.state), 0, True

        x, y = self.state
        if action == 0: x = max(x - 1, 0)
        elif action == 1: x = min(x + 1, self.size - 1)
        elif action == 2: y = max(y - 1, 0)
        elif action == 3: y = min(y + 1, self.size - 1)

        self.state = (x, y)
        done = self.state in self.terminal_states
        return self.state_to_index(self.state), self.reward, done


def choose_action(state, Q1_table, Q2_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(range(4))  
    return np.argmax(Q1_table[state] + Q2_table[state])  

def double_q_learning(env, num_episodes, gamma, alpha, epsilon):
    state_space = env.size * env.size
    Q1_table = np.zeros((state_space, 4)) 
    Q2_table = np.zeros((state_space, 4)) 
    rewards_per_episode = []

    for episode in range(1, num_episodes + 1):
        curr_alpha = alpha(episode) if callable(alpha) else alpha
        curr_epsilon = epsilon(episode) if callable(epsilon) else epsilon
        
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = choose_action(state, Q1_table, Q2_table, curr_epsilon)
            next_state, reward, done = env.step(action)
            

            if np.random.rand() < 0.5:
                
                best_next_action = np.argmax(Q1_table[next_state])
                td_target = reward + gamma * Q2_table[next_state][best_next_action] 
                Q1_table[state][action] += curr_alpha * (td_target - Q1_table[state][action])
            else:
        
                best_next_action = np.argmax(Q2_table[next_state]) 
                td_target = reward + gamma * Q1_table[next_state][best_next_action] 
                Q2_table[state][action] += curr_alpha * (td_target - Q2_table[state][action])
            
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

    return Q1_table, Q2_table, rewards_per_episode

experiments = {
    "Exp_1": {"gamma": 0.99, "alpha": 0.6, "epsilon": 0.6},
    "Exp_2": {"gamma": 0.9, "alpha": 0.1, "epsilon": lambda t: 1/t},
    "Exp_3": {"gamma": 0.6, "alpha": lambda t: 1/t, "epsilon": 0.3},
    "Exp_4": {"gamma": 0.99, "alpha": 0.05, "epsilon": 0.01},
    "Exp_5": {"gamma": 0.6, "alpha": 0.5, "epsilon": 0.9}
}

num_episodes = 10000
results = []

for exp_name, params in experiments.items():
    print(f"Running {exp_name}...")
    env = GridWorld()
    Q1_table, Q2_table, rewards = double_q_learning(
        env, 
        num_episodes=num_episodes,
        gamma=params["gamma"],
        alpha=params["alpha"],
        epsilon=params["epsilon"]
    )
    results.append((exp_name, rewards))

plt.figure(figsize=(12, 6))
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

for i, (name, rewards) in enumerate(results):
    smoothed = moving_average(rewards)
    x_vals = np.arange(len(smoothed)) + 100 
    plt.plot(x_vals, smoothed, 
             color=colors[i], 
             linewidth=1.5, 
             alpha=0.8, 
             label=name)

plt.title('Double Q-Learning Performance Comparison', fontsize=14)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Smoothed Total Reward (100-episode window)', fontsize=12)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
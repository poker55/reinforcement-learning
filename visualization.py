import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import imageio
import io

class GridWorldEnv:
    def __init__(self):
        self.grid_size = 10
        self.state_size = self.grid_size * self.grid_size
        self.action_size = 4  # up, right, down, left
        
        # Create obstacles
        self.grid = np.zeros((self.grid_size, self.grid_size))
        obstacles = [(1, 1), (1, 2), (2, 1), (4, 4), (4, 5), (5, 4),
                    (7, 7), (7, 8), (8, 7), (3, 6), (6, 3)]
        for obs in obstacles:
            self.grid[obs] = 1
            
        self.start_pos = (0, 0)
        self.goal_pos = (9, 9)
        self.current_pos = self.start_pos
        
    def reset(self):
        self.current_pos = self.start_pos
        return self._get_state()
    
    def _get_state(self):
        return self.current_pos[0] * self.grid_size + self.current_pos[1]
    
    def step(self, action):
        x, y = self.current_pos
        done = False
        
        if action == 0:    # up
            new_pos = (max(0, x-1), y)
        elif action == 1:  # right
            new_pos = (x, min(self.grid_size-1, y+1))
        elif action == 2:  # down
            new_pos = (min(self.grid_size-1, x+1), y)
        else:             # left
            new_pos = (x, max(0, y-1))
            
        if self.grid[new_pos] == 1:
            new_pos = self.current_pos
            reward = -2
        else:
            self.current_pos = new_pos
            if new_pos == self.goal_pos:
                reward = 100
                done = True
            else:
                reward = -1
                
        return self._get_state(), reward, done

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.gamma = 0.95
        
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                    self.learning_rate * target
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def create_frame(env, agent, episode):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Environment
    ax1.grid(True)
    ax1.set_xlim(-0.5, env.grid_size-0.5)
    ax1.set_ylim(-0.5, env.grid_size-0.5)
    
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if env.grid[i, j] == 1:
                ax1.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='gray'))
    
    ax1.plot(env.start_pos[1], env.start_pos[0], 'go', markersize=20, label='Start')
    ax1.plot(env.goal_pos[1], env.goal_pos[0], 'ro', markersize=20, label='Goal')
    ax1.plot(env.current_pos[1], env.current_pos[0], 'bo', markersize=15, label='Robot')
    ax1.set_title(f'Environment State (Episode {episode})')
    ax1.legend()

    # Plot 2: Q-value heatmap
    max_q_values = np.max(agent.q_table, axis=1).reshape(env.grid_size, env.grid_size)
    sns.heatmap(max_q_values[::-1], ax=ax2, cmap='YlOrRd', 
                annot=True, fmt='.1f', 
                cbar_kws={'label': 'Max Q-value'})
    ax2.set_yticklabels(range(env.grid_size)[::-1])
    ax2.set_title('Maximum Q-values for each state')
    
    plt.tight_layout()
    
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return buf

# Training with frame collection
env = GridWorldEnv()
agent = QLearningAgent(env.state_size, env.action_size)
episodes = 1000
max_steps = 200

# Collect frames
frames = []
selected_episodes = list(range(0, 100, 10)) + list(range(100, episodes + 1, 50))  # Capture more frames early on

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    if episode in selected_episodes:
        print(f"Capturing frame for episode {episode}")
        frame = create_frame(env, agent, episode)
        frames.append(imageio.imread(frame))

# Save the animation
print("Creating GIF...")
imageio.mimsave('q_learning_training.gif', frames, fps=2)
print("Training complete! Check q_learning_training.gif for the visualization.")
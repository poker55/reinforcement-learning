import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class GridWorldEnv:
    def __init__(self):
        self.grid_size = 10
        self.state_size = self.grid_size * self.grid_size
        self.action_size = 4  # up, right, down, left
        
        # Create obstacles (1 represents obstacle, 0 represents free space)
        self.grid = np.zeros((self.grid_size, self.grid_size))
        # Set some obstacles
        obstacles = [(1, 1), (1, 2), (2, 1), (4, 4), (4, 5), (5, 4),
                    (7, 7), (7, 8), (8, 7), (3, 6), (6, 3)]
        for obs in obstacles:
            self.grid[obs] = 1
            
        # Set start and goal positions
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
        
        # Define movement for each action
        if action == 0:    # up
            new_pos = (max(0, x-1), y)
        elif action == 1:  # right
            new_pos = (x, min(self.grid_size-1, y+1))
        elif action == 2:  # down
            new_pos = (min(self.grid_size-1, x+1), y)
        else:             # left
            new_pos = (x, max(0, y-1))
            
        # Check if new position is valid (not an obstacle)
        if self.grid[new_pos] == 1:
            new_pos = self.current_pos
            reward = -2  # Penalty for hitting obstacle
        else:
            self.current_pos = new_pos
            
            if new_pos == self.goal_pos:
                reward = 100  # Reward for reaching goal
                done = True
            else:
                reward = -1  # Small penalty for each move
                done = False
                
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

def visualize_episode(env, agent, episode):
    plt.clf()
    plt.figure(figsize=(8, 8))
    
    # Draw grid
    plt.grid(True)
    plt.xlim(-0.5, env.grid_size-0.5)
    plt.ylim(-0.5, env.grid_size-0.5)
    
    # Draw obstacles
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if env.grid[i, j] == 1:
                plt.gca().add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='gray'))
    
    # Draw start and goal
    plt.plot(env.start_pos[1], env.start_pos[0], 'go', markersize=20, label='Start')
    plt.plot(env.goal_pos[1], env.goal_pos[0], 'ro', markersize=20, label='Goal')
    
    # Draw current position
    plt.plot(env.current_pos[1], env.current_pos[0], 'bo', markersize=15, label='Robot')
    
    plt.title(f'Episode {episode}')
    plt.legend()
    plt.show()

# Training
env = GridWorldEnv()
agent = QLearningAgent(env.state_size, env.action_size)
episodes = 1000
max_steps = 200

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        
        if episode % 100 == 0 and step % 10 == 0:
            visualize_episode(env, agent, episode)
        
        if done:
            break
    
    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# Test the trained agent
state = env.reset()
done = False
while not done:
    action = agent.get_action(state)
    state, reward, done = env.step(action)
    visualize_episode(env, agent, "Final Path")
    plt.pause(0.5)  # Add delay to visualize movement
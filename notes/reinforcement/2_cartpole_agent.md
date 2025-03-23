# Transforming `cartpole.py` into a Learning Agent

![CartPole](./2_cartpole_agent.gif)

_In this image, you would notice that the cartpole is now learning to balance itself than without the agent._

To convert your `cartpole.py` script into a learning agent capable of mastering the CartPole task, we'll follow the roadmap you've outlined. We'll implement a **Deep Q-Network (DQN)** approach, which combines Q-Learning with deep neural networks to handle complex state spaces.

Here's a step-by-step guide to achieve this transformation:

---

## Table of Contents

1. [Overview](#overview)
2. [Code](#code)
3. [Project Structure](#project-structure)
4. [Step 1: Install Dependencies](#step-1-install-dependencies)
5. [Step 2: Implement the Neural Network](#step-2-implement-the-neural-network)
6. [Step 3: Create the Replay Buffer](#step-3-create-the-replay-buffer)
7. [Step 4: Develop the DQN Agent](#step-4-develop-the-dqn-agent)
8. [Step 5: Modify `cartpole.py`](#step-5-modify-cartpolepy)
9. [Step 6: Training and Evaluation](#step-6-training-and-evaluation)
10. [Optimization Techniques](#optimization-techniques)
11. [Conclusion](#conclusion)

---

## Overview

We'll enhance your `cartpole.py` script by integrating a DQN agent. This involves creating additional modules for the neural network, replay buffer, and the agent itself. The main script will be modified to utilize these components for training and decision-making.

---

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Dueling streams
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage into Q-values
        q_vals = value + (advantage - advantage.mean())
        return q_vals

import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=128,
        lr=1e-3,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        target_update_freq=1000
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=buffer_size)
    
    def select_action(self, state):
        self.steps_done += 1
        # Epsilon decay
        self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
                       np.exp(-1. * self.steps_done / self.epsilon_decay)
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def push_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Double DQN: Use policy_net to select the best action, then use target_net to evaluate it
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
        
        # Expected Q values
        expected_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q, expected_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

import gymnasium as gym
import pygame
import sys
import numpy as np
import torch

# Initialize Pygame and environment
def initialize_game():
    pygame.init()
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env

# Set up Pygame display
def setup_display():
    env_width, env_height = 800, 600
    stats_width = 400
    screen_width, screen_height = env_width + stats_width, env_height
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("CartPole-v1 with Info Overlay")
    return screen, env_width, stats_width, screen_height

# Function to render text on the Pygame window
def render_text(screen, text, position, font_size=24, color=(255, 255, 255)):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

# Function to draw a semi-transparent background
def draw_transparent_box(screen, position, size, color=(0, 0, 0), alpha=128):
    s = pygame.Surface(size, pygame.SRCALPHA)
    s.fill((*color, alpha))
    screen.blit(s, position)

# Render game state
def render_game_state(screen, env_image, env_width, stats_width, screen_height, episode, step, action, reward, cumulative_reward, next_state, done):
    # Render the environment
    env_surface = pygame.surfarray.make_surface(env_image.swapaxes(0, 1))
    screen.blit(env_surface, (0, 0))

    # Draw semi-transparent background for stats on the right side
    draw_transparent_box(screen, (env_width, 0), (stats_width, screen_height), color=(0, 0, 0), alpha=180)

    # Render stats on the right side
    render_text(screen, f"Episode: {episode + 1}", (env_width + 20, 20))
    render_text(screen, f"Step: {step}", (env_width + 20, 60))
    render_text(screen, f"Action: {action} ({'Left' if action == 0 else 'Right'})", (env_width + 20, 100))
    render_text(screen, f"Reward: {reward:.2f}", (env_width + 20, 140))
    render_text(screen, f"Cumulative Reward: {cumulative_reward:.2f}", (env_width + 20, 180))

    # Display state information
    render_text(screen, "State:", (env_width + 20, 230))
    render_text(screen, f"  Cart Position: {next_state[0]:.4f}", (env_width + 20, 270))
    render_text(screen, f"  Cart Velocity: {next_state[1]:.4f}", (env_width + 20, 310))
    render_text(screen, f"  Pole Angle: {next_state[2]:.4f} rad ({np.degrees(next_state[2]):.2f}°)", (env_width + 20, 350))
    render_text(screen, f"  Pole Angular Velocity: {next_state[3]:.4f}", (env_width + 20, 390))

    # Display termination conditions
    render_text(screen, "Termination Conditions:", (env_width + 20, 440))
    render_text(screen, f"  |Cart Position| < 2.4: {abs(next_state[0]) < 2.4}", (env_width + 20, 480))
    render_text(screen, f"  |Pole Angle| < 12°: {abs(np.degrees(next_state[2])) < 12}", (env_width + 20, 520))

    if done:
        reason = "Pole fell or cart out of bounds" if isinstance(done, bool) else "Max steps reached"
        render_text(screen, f"Episode ended: {reason}", (env_width + 20, 560), color=(255, 0, 0))

    # Update the full display
    pygame.display.flip()

# Modified run_episode to handle training
def run_episode(env, screen, env_width, stats_width, screen_height, episode, agent):
    state, _ = env.reset()
    done = False
    cumulative_reward = 0
    step = 0

    while not done:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        agent.push_memory(state, action, reward, next_state, done)
        agent.optimize_model()

        cumulative_reward += reward
        step += 1

        # Render the environment
        env_image = env.render()

        render_game_state(screen, env_image, env_width, stats_width, screen_height, episode, step, action, reward, cumulative_reward, next_state, done)

        state = next_state

        # Update target network periodically
        if agent.steps_done % agent.target_update_freq == 0:
            agent.update_target_network()

    return cumulative_reward

import os

# Main function
def main():
    env = initialize_game()
    screen, env_width, stats_width, screen_height = setup_display()
    clock = pygame.time.Clock()
    fps = 60  # Increased FPS for smoother training

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=128,
        lr=1e-3,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        target_update_freq=1000
    )
    
    # Before training loop
    # agent.load_model("models/dqn_cartpole_episode_1000.pth")
    
    # Create the models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    num_episodes = 1000
    for episode in range(num_episodes):
        episode_reward = run_episode(env, screen, env_width, stats_width, screen_height, episode, agent)

        if episode_reward is None:  # User closed the window
            break

        # Short pause between episodes
        # pygame.time.wait(100)

        # Log progress
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

        # After logging
        if (episode + 1) % 100 == 0:
            model_path = f"models/dqn_cartpole_episode_{episode + 1}.pth"
            agent.save_model(model_path)
            print(f"Model saved at: {model_path}")

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
```

---

## Project Structure

To maintain organization, we'll structure the project as follows:

```
project/
├── scripts/
│   ├── cartpole.py
│   ├── agent.py
│   ├── network.py
│   └── replay_buffer.py
├── models/
│   └── (saved models will be stored here)
└── requirements.txt
```


---

## Step 1: Install Dependencies

Ensure you have the required libraries installed. You can create a `requirements.txt` for easy installation.

```
gymnasium
pygame
numpy
torch
```

Install the dependencies using pip:

```
pip install -r requirements.txt
```

---

## Step 2: Implement the Neural Network

Create a neural network to approximate the Q-values.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
```


### Explanation

- **DQNNetwork**: A simple feedforward neural network with two hidden layers using ReLU activation.
- **Inputs**: `state_size` (number of state features) and `action_size` (number of possible actions).
- **Output**: Q-values for each action.

---

## Step 3: Create the Replay Buffer

Implement experience replay to store and sample experiences.

```python
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
```


### Explanation

- **ReplayBuffer**: Stores experiences as tuples of `(state, action, reward, next_state, done)`.
- **Capacity**: Maximum number of experiences to store.
- **Sampling**: Randomly samples a batch of experiences for training.

---

## Step 4: Develop the DQN Agent

Create the agent that interacts with the environment and learns from experiences.

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from network import DQNNetwork
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=128,
        lr=1e-3,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        target_update_freq=1000
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=buffer_size)
    
    def select_action(self, state):
        self.steps_done += 1
        # Epsilon decay
        self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
                       np.exp(-1. * self.steps_done / self.epsilon_decay)
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def push_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        # Expected Q values
        expected_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q, expected_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
```


### Explanation

- **DQNAgent**: Handles action selection, experience storage, and model optimization.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation with decaying epsilon.
- **Memory**: Utilizes the `ReplayBuffer` to store and sample experiences.
- **Optimization**: Updates the policy network using sampled experiences.
- **Target Network**: Periodically updated to stabilize training.

---

## Step 5: Modify `cartpole.py`

Integrate the DQN agent into your main script by replacing random actions with policy-derived actions and incorporating the training loop.

```python
import gymnasium as gym
import pygame
import sys
import numpy as np
import torch

from agent import DQNAgent

# Initialize Pygame and environment
def initialize_game():
    pygame.init()
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env

# Set up Pygame display
def setup_display():
    env_width, env_height = 800, 600
    stats_width = 400
    screen_width, screen_height = env_width + stats_width, env_height
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("CartPole-v1 with Info Overlay")
    return screen, env_width, stats_width, screen_height

# Function to render text on the Pygame window
def render_text(screen, text, position, font_size=24, color=(255, 255, 255)):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

# Function to draw a semi-transparent background
def draw_transparent_box(screen, position, size, color=(0, 0, 0), alpha=128):
    s = pygame.Surface(size, pygame.SRCALPHA)
    s.fill((*color, alpha))
    screen.blit(s, position)

# Render game state
def render_game_state(screen, env_image, env_width, stats_width, screen_height, episode, step, action, reward, cumulative_reward, next_state, done):
    # Render the environment
    env_surface = pygame.surfarray.make_surface(env_image.swapaxes(0, 1))
    screen.blit(env_surface, (0, 0))

    # Draw semi-transparent background for stats on the right side
    draw_transparent_box(screen, (env_width, 0), (stats_width, screen_height), color=(0, 0, 0), alpha=180)

    # Render stats on the right side
    render_text(screen, f"Episode: {episode + 1}", (env_width + 20, 20))
    render_text(screen, f"Step: {step}", (env_width + 20, 60))
    render_text(screen, f"Action: {action} ({'Left' if action == 0 else 'Right'})", (env_width + 20, 100))
    render_text(screen, f"Reward: {reward:.2f}", (env_width + 20, 140))
    render_text(screen, f"Cumulative Reward: {cumulative_reward:.2f}", (env_width + 20, 180))

    # Display state information
    render_text(screen, "State:", (env_width + 20, 230))
    render_text(screen, f"  Cart Position: {next_state[0]:.4f}", (env_width + 20, 270))
    render_text(screen, f"  Cart Velocity: {next_state[1]:.4f}", (env_width + 20, 310))
    render_text(screen, f"  Pole Angle: {next_state[2]:.4f} rad ({np.degrees(next_state[2]):.2f}°)", (env_width + 20, 350))
    render_text(screen, f"  Pole Angular Velocity: {next_state[3]:.4f}", (env_width + 20, 390))

    # Display termination conditions
    render_text(screen, "Termination Conditions:", (env_width + 20, 440))
    render_text(screen, f"  |Cart Position| < 2.4: {abs(next_state[0]) < 2.4}", (env_width + 20, 480))
    render_text(screen, f"  |Pole Angle| < 12°: {abs(np.degrees(next_state[2])) < 12}", (env_width + 20, 520))

    if done:
        reason = "Pole fell or cart out of bounds" if isinstance(done, bool) else "Max steps reached"
        render_text(screen, f"Episode ended: {reason}", (env_width + 20, 560), color=(255, 0, 0))

    # Update the full display
    pygame.display.flip()

# Modified run_episode to handle training
def run_episode(env, screen, env_width, stats_width, screen_height, episode, agent):
    state, _ = env.reset()
    done = False
    cumulative_reward = 0
    step = 0

    while not done:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        agent.push_memory(state, action, reward, next_state, done)
        agent.optimize_model()

        cumulative_reward += reward
        step += 1

        # Render the environment
        env_image = env.render()

        render_game_state(screen, env_image, env_width, stats_width, screen_height, episode, step, action, reward, cumulative_reward, next_state, done)

        state = next_state

        # Update target network periodically
        if agent.steps_done % agent.target_update_freq == 0:
            agent.update_target_network()

    return cumulative_reward

# Main function
def main():
    env = initialize_game()
    screen, env_width, stats_width, screen_height = setup_display()
    clock = pygame.time.Clock()
    fps = 60  # Increased FPS for smoother training

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=128,
        lr=1e-3,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        target_update_freq=1000
    )
    
    num_episodes = 1000
    for episode in range(num_episodes):
        episode_reward = run_episode(env, screen, env_width, stats_width, screen_height, episode, agent)

        if episode_reward is None:  # User closed the window
            break

        # Short pause between episodes
        pygame.time.wait(100)

        # Log progress
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
```


### Explanation

- **Agent Integration**: Replaces random action selection with `agent.select_action(state)`.
- **Experience Storage**: Stores experiences using `agent.push_memory(...)`.
- **Model Optimization**: Calls `agent.optimize_model()` after each step.
- **Target Network Update**: Periodically updates the target network for stability.
- **Logging**: Prints cumulative rewards for each episode to monitor training progress.
- **Increased Episodes**: Set to 1000 for ample training opportunities.

---

## Step 6: Training and Evaluation

### 6.1 Training the Agent

Run the modified `cartpole.py` script to start training:

```bash
python scripts/cartpole.py
```
As training progresses, you should observe the cumulative rewards increasing, indicating that the agent is learning to balance the pole more effectively.

### 6.2 Monitoring Performance

Monitor the printed rewards in the console to assess the agent's performance. Optionally, you can implement more sophisticated logging (e.g., plotting rewards over time) for better visualization.

### 6.3 Saving the Model

To save the trained model for later use:

1. **Modify `agent.py` to include a save method:**

    ```python
        def save_model(self, filepath):
            torch.save(self.policy_net.state_dict(), filepath)
    ```

2. **Update `cartpole.py` to save the model periodically:**

    ```python
        # After logging
        if (episode + 1) % 100 == 0:
            agent.save_model(f"models/dqn_cartpole_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")
    ```

### 6.4 Loading a Saved Model

To load a saved model for evaluation or further training:

1. **Add a load method in `agent.py`:**

    ```python
        def load_model(self, filepath):
            self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
    ```

2. **Use the load method in `cartpole.py`:**

    ```python
        # Before training loop
        # agent.load_model("models/dqn_cartpole_episode_1000.pth")
    ```

---

## Optimization Techniques

To further enhance your agent's performance and training stability, consider implementing the following optimization techniques:

### 8.1 Double DQN

Double DQN mitigates overestimation of Q-values by decoupling action selection and evaluation.

**Implementation:**

Modify the `optimize_model` method in `agent.py`:

```python
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Double DQN: Use policy_net to select the best action, then use target_net to evaluate it
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
        
        # Expected Q values
        expected_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q, expected_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```


### 8.2 Prioritized Experience Replay

Prioritized Experience Replay samples more important transitions more frequently, improving learning efficiency.

**Implementation:**

Implementing prioritized replay is more involved and would require modifying the `ReplayBuffer` to support sampling based on priority. Consider using existing libraries or resources for guidance.

### 8.3 Dueling Networks

Dueling Networks separately estimate state-value and advantage, enhancing learning.

**Implementation:**

Modify the `DQNNetwork` to include separate streams for value and advantage:

```python
    class DQNNetwork(nn.Module):
        def __init__(self, state_size, action_size, hidden_size=128):
            super(DQNNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            
            # Dueling streams
            self.value_stream = nn.Linear(hidden_size, 1)
            self.advantage_stream = nn.Linear(hidden_size, action_size)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            
            # Combine value and advantage into Q-values
            q_vals = value + (advantage - advantage.mean())
            return q_vals
```


---

## Conclusion

By following the steps outlined above, you've successfully transformed your `cartpole.py` script into a robust learning agent using Deep Q-Networks. The agent can now learn to balance the pole through interaction with the environment, leveraging experience replay and neural network approximation.

### Next Steps

- **Hyperparameter Tuning**: Experiment with different hyperparameters like learning rate, batch size, and epsilon decay to optimize performance.
- **Advanced Techniques**: Implement prioritized experience replay, dueling networks, or other advanced methods to further enhance learning.
- **Logging and Visualization**: Incorporate logging libraries or visualization tools to better monitor training progress and agent behavior.
- **Scalability**: Adapt the agent to more complex environments or tasks beyond CartPole.

Happy coding and training!
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
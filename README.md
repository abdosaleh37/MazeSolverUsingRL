# MazeWithReinforcementLearning

A Python project for training and visualizing reinforcement learning agents (Q-learning and Policy Gradient) to solve a maze. Features a modern GUI for agent selection, smooth maze visualization, and robust, tunable RL agents.

---

## Features

- **Two RL Agents:** Q-learning and Policy Gradient (PG) agents, each with tunable hyperparameters.
- **Modern GUI:** Agent selection window with background image, rounded buttons, and clear layout.
- **Smooth Visualization:** Maze rendering using Pygame for interactive, flicker-free display.
- **Training Curves:** Plots for both reward and steps per episode.
- **Stable Learning:** Policy Gradient agent with entropy regularization, learning rate decay, and reward normalization.
- **Easy Customization:** Adjust maze size, agent parameters, and visualization options.

---

## Installation


1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Typical requirements:*
   - numpy
   - matplotlib
   - pygame
   - ttkbootstrap (for modern Tkinter GUI)

---

## Usage

1. **Run the main program:**
   ```bash
   python main.py
   ```

2. **Select Agent:**
   - Choose between Q-learning and Policy Gradient in the GUI.
   - Adjust hyperparameters if desired.

3. **Train the Agent:**
   - The agent will train on the maze environment.
   - Training curves (reward and steps) will be displayed.

4. **Visualize the Solution:**
   - After training, the Pygame window will open to show the agent solving the maze.


## Customization

- **Maze Size:** Change in `maze_env.py`.

---

## Troubleshooting

- **Flickering/Slow Visualization:** Ensure Pygame is installed and used for rendering.
- **Agent Stuck or Poor Learning:** Tune learning rate, entropy coefficient, and other hyperparameters in the GUI or agent code.
- **GUI Issues:** Make sure `ttkbootstrap` is installed.

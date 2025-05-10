import pygame

class TrainingPlotter:
    def __init__(self, width=1200, height=400, title="Training Progress"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        
        # Colors
        self.colors = {
            'background': (20, 20, 40),
            'grid': (40, 40, 60),
            'reward': (0, 255, 0),
            'steps': (0, 191, 255),  # Deep sky blue
            'text': (200, 200, 200)
        }
        
        # Font
        self.font = pygame.font.Font(None, 24)
        
        # Data storage
        self.rewards = []
        self.steps = []
        self.min_reward = 0
        self.min_steps = 0
        self.max_reward = 0
        self.max_steps = 0
        
        # Padding and margins
        self.padding = 50
        self.plot_width = (width - 3 * self.padding) // 2  # Half width for each plot
        self.plot_height = height - 2 * self.padding

    def update(self, reward=None, steps=None):
        if reward is not None:
            self.rewards.append(reward)
            self.max_reward = max(self.max_reward, reward)
            self.min_reward = min(self.min_reward, reward)
        if steps is not None:
            self.steps.append(steps)
            self.max_steps = max(self.max_steps, steps)
            self.min_steps = min(self.min_steps, steps)

    def draw_grid(self, x_offset):
        for i in range(5):
            y = self.padding + (i * self.plot_height) // 4
            pygame.draw.line(self.screen, self.colors['grid'],
                            (x_offset, y),
                            (x_offset + self.plot_width, y))

            # Y-axis for rewards
            if x_offset == self.padding and self.max_reward > self.min_reward:
                value = self.min_reward + (self.max_reward - self.min_reward) * (4 - i) / 4
                text = self.font.render(f"{value:.1f}", True, self.colors['text'])
                self.screen.blit(text, (x_offset - 40, y - 10))

            # Y-axis for steps
            elif x_offset == 2 * self.padding + self.plot_width and self.max_steps > self.min_steps:
                value = self.min_steps + (self.max_steps - self.min_steps) * (4 - i) / 4
                text = self.font.render(f"{int(value)}", True, self.colors['text'])
                self.screen.blit(text, (x_offset - 40, y - 10))

        # X-axis ticks (episodes)
        data = self.rewards if x_offset == self.padding else self.steps
        num_points = len(data)
        if num_points > 1:
            for i in range(0, num_points, max(1, num_points // 5)):
                x = x_offset + (i * self.plot_width) / (num_points - 1)
                pygame.draw.line(self.screen, self.colors['grid'], (x, self.padding + self.plot_height),
                                (x, self.padding + self.plot_height + 5))
                text = self.font.render(str(i), True, self.colors['text'])
                self.screen.blit(text, (x - 10, self.padding + self.plot_height + 10))


    def draw_curves(self):
        if not self.rewards and not self.steps:
            return

        # Draw reward curve
        if self.rewards:
            points = []
            min_reward = min(self.rewards)
            reward_range = self.max_reward - min_reward or 1
            for i, reward in enumerate(self.rewards):
                x = self.padding + (i * self.plot_width) / max(1, len(self.rewards) - 1)
                y = self.height - self.padding - ((reward - min_reward) * self.plot_height) / reward_range
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.colors['reward'], False, points, 2)

        # Draw steps curve
        if self.steps:
            points = []
            for i, steps in enumerate(self.steps):
                x = 2 * self.padding + self.plot_width + (i * self.plot_width) / max(1, len(self.steps) - 1)
                y = self.height - self.padding - (steps * self.plot_height) / max(1, self.max_steps)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.colors['steps'], False, points, 2)

    def draw_labels(self):
        # Draw background
        self.screen.fill(self.colors['background'])
        
        # Draw reward plot label
        reward_text = self.font.render("Rewards", True, self.colors['reward'])
        self.screen.blit(reward_text, (self.padding + self.plot_width//2 - 30, 20))
        
        # Draw steps plot label
        steps_text = self.font.render("Steps", True, self.colors['steps'])
        self.screen.blit(steps_text, (2 * self.padding + self.plot_width + self.plot_width//2 - 20, 20))

    def render(self):
        self.draw_labels()
        self.draw_grid(self.padding)  # Grid for rewards
        self.draw_grid(2 * self.padding + self.plot_width)  # Grid for steps
        self.draw_curves()
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        return True

def plot_learning_curve(rewards, steps, agent_type="Training Progress", delay_ms=100):
    """
    Plot learning curves using Pygame for real-time visualization with a delay.
    
    Args:
        rewards (list): List of reward values
        steps (list): List of steps per episode
        agent_type (str): Type of agent (e.g., "Q Learning" or "Policy Gradient")
        delay_ms (int): Delay in milliseconds between frame updates
    """
    plotter = TrainingPlotter(title=f"{agent_type} Training Progress - To show Maze click X or wait 10 seconds")
    
    for reward, step in zip(rewards, steps):
        plotter.update(reward, step)
        plotter.render()
        if not plotter.handle_events():
            return
        pygame.time.wait(delay_ms)  # Add delay between frames

    # Keep the window open until the user closes it or wait 10 seconds
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < 10000:  # 10,000 ms = 10 seconds
        if not plotter.handle_events():
            break
        pygame.time.wait(100)
    pygame.quit()


def plot_rewards(rewards, agent_type="Training Progress"):
    """
    Plot only rewards using Pygame for real-time visualization.
    
    Args:
        rewards (list): List of reward values
        agent_type (str): Type of agent (e.g., "Q Learning" or "Policy Gradient")
    """
    plot_learning_curve(rewards, [], agent_type=agent_type)

def plot_steps(steps, agent_type="Training Progress"):
    """
    Plot only steps using Pygame for real-time visualization.
    
    Args:
        steps (list): List of steps per episode
        agent_type (str): Type of agent (e.g., "Q Learning" or "Policy Gradient")
    """
    plot_learning_curve([], steps, agent_type=agent_type) 
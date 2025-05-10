import pygame
import numpy as np

class PygameMazeRenderer:
    def __init__(self, maze_size, cell_size=40):
        """
        Initialize the Pygame maze renderer.
        
        Parameters:
        - maze_size (tuple): Size of the maze (rows, cols)
        - cell_size (int): Size of each cell in pixels
        """
        self.cell_size = cell_size
        self.maze_size = maze_size
        self.width = maze_size[1] * cell_size
        self.height = maze_size[0] * cell_size
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Maze Environment")
        
        # Colors
        self.colors = {
            'wall': (50, 50, 50),      # Dark gray
            'path': (255, 255, 255),   # White
            'agent': (0, 120, 255),    # Blue
            'goal': (255, 215, 0),     # Gold
            'grid': (200, 200, 200)    # Light gray
        }
        
        # Animation properties
        self.animation_speed = 0.1  # seconds per frame
        self.last_update = 0
        
    def render(self, maze, agent_pos, goal_pos):
        """
        Render the current state of the maze.
        
        Parameters:
        - maze (numpy.ndarray): The maze grid
        - agent_pos (tuple): Current position of the agent
        - goal_pos (tuple): Position of the goal
        """
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        # Clear the screen
        self.screen.fill(self.colors['path'])
        
        # Draw the maze
        for i in range(self.maze_size[0]):
            for j in range(self.maze_size[1]):
                rect = pygame.Rect(
                    j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                if maze[i, j] == 1:  # Wall
                    pygame.draw.rect(self.screen, self.colors['wall'], rect)
                else:  # Path
                    pygame.draw.rect(self.screen, self.colors['path'], rect)
                    pygame.draw.rect(self.screen, self.colors['grid'], rect, 1)
        
        # Draw the goal
        goal_rect = pygame.Rect(
            goal_pos[1] * self.cell_size,
            goal_pos[0] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.colors['goal'], goal_rect)
        
        # Draw the agent
        agent_rect = pygame.Rect(
            agent_pos[1] * self.cell_size + self.cell_size * 0.2,
            agent_pos[0] * self.cell_size + self.cell_size * 0.2,
            self.cell_size * 0.6,
            self.cell_size * 0.6
        )
        pygame.draw.ellipse(self.screen, self.colors['agent'], agent_rect)
        
        # Update the display
        pygame.display.flip()
        
        # Control animation speed
        current_time = pygame.time.get_ticks()
        if current_time - self.last_update < self.animation_speed * 1000:
            pygame.time.wait(int(self.animation_speed * 1000 - (current_time - self.last_update)))
        self.last_update = pygame.time.get_ticks()
        
        return True
    
    def close(self):
        """Close the Pygame window."""
        pygame.quit() 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def generate_maze(size=(10, 10), wall_prob=0.4, max_attempts=100):
    """
    Generate a random maze with a guaranteed path from start to goal.

    Parameters:
    - size: Dimensions of the maze (rows, columns).
    - wall_prob: Probability of placing a wall at each cell.
    - max_attempts: Maximum number of retries to generate a solvable maze.

    Returns:
    - maze: A maze grid where 0 = free space, 1 = wall.
    """
    maze = np.zeros(size)
    
    # Randomly place walls based on the given probability
    for i in range(size[0]):
        for j in range(size[1]):
            if np.random.rand() < wall_prob:
                maze[i, j] = 1

    # Ensure start and goal positions are not blocked
    maze[0, 0] = 0
    maze[size[0]-1, size[1]-1] = 0

    # Try regenerating until a valid path exists or max_attempts reached
    attempts = 0
    while attempts < max_attempts:
        if has_path(maze, (0, 0), (size[0]-1, size[1]-1)):
            return maze
        else:
            # Recreate maze if not solvable
            maze = np.zeros(size)
            for i in range(size[0]):
                for j in range(size[1]):
                    if np.random.rand() < wall_prob:
                        maze[i, j] = 1
            maze[0, 0] = 0
            maze[size[0]-1, size[1]-1] = 0
            attempts += 1

    print("Failed to generate a maze with a valid path.")
    return maze


# Check if there is a valid path from start to goal using DFS.
def has_path(maze, start, goal, visited=None):
    if visited is None:
        visited = set()
    
    if start == goal:
        return True

    if (start[0] < 0 or start[1] < 0 or start[0] >= len(maze) or start[1] >= len(maze[0]) or
        maze[start[0], start[1]] == 1):
        return False

    if start in visited:
        return False

    visited.add(start)

    # Explore 4 directions (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for direction in directions:
        next_position = (start[0] + direction[0], start[1] + direction[1])
        if has_path(maze, next_position, goal, visited):
            return True

    return False

# Render the current maze environment using matplotlib.
def render_maze(maze, agent_pos, goal_pos):
    grid = np.copy(maze)

    gx, gy = goal_pos
    grid[gx, gy] = 2    # Mark the goal with a different value

    # Define color map: 0 = wall (dark), 1 = path (white), 2 = goal (yellow)
    colors = ['dimgray', 'white', 'gold']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure("Maze View")
    plt.clf()
    plt.imshow(grid, cmap=cmap, norm=norm)

    # Draw grid lines
    rows, cols = grid.shape
    for x in range(rows + 1):
        plt.axhline(x - 0.5, color='gray', linewidth=0.3)
    for y in range(cols + 1):
        plt.axvline(y - 0.5, color='gray', linewidth=0.3)

    plt.xticks([])
    plt.yticks([])

    # Draw agent as a blue circle
    ax, ay = agent_pos
    circle = plt.Circle((ay, ax), 0.3, color='blue', ec='black', lw=1)
    plt.gca().add_patch(circle)

    plt.gca().set_aspect('equal')
    plt.pause(0.1)


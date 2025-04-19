import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def generate_maze(size=(10, 10), wall_prob=0.5):
    maze = np.zeros(size, dtype=int)
    for i in range(size[0]):
        for j in range(size[1]):
            if np.random.rand() < wall_prob:
                maze[i, j] = 1  # 1 = wall
    maze[0, 0] = 0  # Start
    maze[-1, -1] = 0  # Goal
    if not has_path(maze, (0, 0), (size[0] - 1, size[1] - 1)):
        return generate_maze(size, wall_prob)  # regenerate if there is no path to goal

    return maze

def has_path(maze, start, goal):
    # خوارزمية DFS (البحث بالعمق) للبحث عن المسار
    stack = [start]
    visited = set()

    while stack:
        x, y = stack.pop()

        if (x, y) == goal:
            return True

        if (x, y) not in visited:
            visited.add((x, y))

            # تحركات الوكيل: UP, DOWN, LEFT, RIGHT
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy

                if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx, ny] == 0:
                    stack.append((nx, ny))

    return False

def render_maze(maze, agent_pos, goal_pos):
    grid = np.copy(maze)

    # goal position
    gx, gy = goal_pos
    grid[gy, gx] = 2  # plotting goal

    # determine colors for walls, path and goal
    colors = ['white', 'lightgray', 'green'] 
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.xticks([]); plt.yticks([])

    # رسم الوكيل (دائرة زرقاء)
    ax, ay = agent_pos
    circle = plt.Circle((ay, ax), 0.3, color='blue')
    plt.gca().add_patch(circle)

    plt.pause(0.1)
    plt.clf()


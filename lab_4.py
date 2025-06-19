"""Grid Maze
A 2D grid with:

0 → open cell

1 → wall

You start at (0, 0) and want to reach (n-1, m-1)"""

#Grid Example

maze = [
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 0],
    [1, 1, 1, 0]
]

#Breadth-First Search (BFS)

from collections import deque

def bfs(maze):
    rows, cols = len(maze), len(maze[0])
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    visited = set()
    queue = deque([(start, [start])])  # (current_cell, path_to_cell)

    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:  # up, down, left, right
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
    return None


#Depth-First Search (DFS)

def dfs(maze):
    rows, cols = len(maze), len(maze[0])
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    visited = set()
    stack = [(start, [start])]

    while stack:
        (x, y), path = stack.pop()
        if (x, y) == goal:
            return path

        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0:
                stack.append(((nx, ny), path + [(nx, ny)]))
    return None



#Test Code

maze = [
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 0],
    [1, 1, 1, 0]
]

print("BFS Path:", bfs(maze))
print("DFS Path:", dfs(maze))


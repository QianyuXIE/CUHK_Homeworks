# CSC5010 AI Homework 1
# A1: 8-Puzzle Problem (DFS)
# Xie Qianyu
# 225040249@link.cuhk.edu.cn

def get_neighbors(state):
    neighbors = []
    idx = state.index(0)
    r, c = idx // 3, idx % 3
    # 移动顺序: Up, Down, Left, Right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_state = list(state)
            target = nr * 3 + nc
            new_state[idx], new_state[target] = new_state[target], new_state[idx]
            neighbors.append(tuple(new_state))
    return neighbors

def dfs(start, goal, limit=20):
    stack = [(start, [start])]
    visited = {start: 0}
    
    while stack:
        (state, path) = stack.pop()
        if state == goal:
            return path
        
        if len(path) <= limit:
            for neighbor in get_neighbors(state):
                if neighbor not in visited or visited[neighbor] > len(path):
                    visited[neighbor] = len(path)
                    stack.append((neighbor, path + [neighbor]))
    return None

def print_puzzle(state):
    for i in range(0, 9, 3):
        print(f"{state[i]} {state[i+1]} {state[i+2]}")

# 作业给定状态
START = (2, 8, 3, 1, 6, 4, 7, 0, 5)
GOAL = (1, 2, 3, 8, 0, 4, 7, 6, 5)

print("Running Task A1: DFS...")
result = dfs(START, GOAL)
if result:
    print(f"Moves: {len(result)-1}")
    for step in result:
        print_puzzle(step)
        print("---")
# CSC5010 AI Homework 1
# B1: 8-Puzzle Problem (A* Search)
# Xie Qianyu
# 225040249@link.cuhk.edu.cn

import heapq

def manhattan_distance(state, goal):
    distance = 0
    for i in range(9):
        if state[i] != 0:
            g_idx = goal.index(state[i])
            distance += abs(i // 3 - g_idx // 3) + abs(i % 3 - g_idx % 3)
    return distance

def a_star(start, goal):
    # (f, g, current_state, path)
    open_set = [(manhattan_distance(start, goal), 0, start, [start])]
    visited = {start: 0}
    
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        
        if current == goal:
            return path
        
        idx = current.index(0)
        r, c = idx // 3, idx % 3
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                neighbor = list(current)
                target = nr * 3 + nc
                neighbor[idx], neighbor[target] = neighbor[target], neighbor[idx]
                neighbor = tuple(neighbor)
                
                new_g = g + 1
                if neighbor not in visited or new_g < visited[neighbor]:
                    visited[neighbor] = new_g
                    new_f = new_g + manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))
    return None

def print_puzzle(state):
    for i in range(0, 9, 3):
        print(f"{state[i]} {state[i+1]} {state[i+2]}")

START = (2, 8, 3, 1, 6, 4, 7, 0, 5)
GOAL = (1, 2, 3, 8, 0, 4, 7, 6, 5)

print("Running Task B1: A*...")
result = a_star(START, GOAL)
if result:
    print(f"Moves: {len(result)-1}")
    for step in result:
        print_puzzle(step)
        print("---")
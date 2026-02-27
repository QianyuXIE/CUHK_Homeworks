# CSC5010 AI Homework 1
# B2: 8-Puzzle Problem (A*)
# Xie Qianyu
# 225040249@link.cuhk.edu.cn

import random
import heapq


STUDENT_ID = 225040249
random.seed(STUDENT_ID)
GOAL_STATE = (1, 2, 3, 8, 0, 4, 7, 6, 5)

def get_neighbors(state):
    neighbors = []
    idx = state.index(0)
    r, c = idx // 3, idx % 3
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            n_list = list(state)
            target = nr * 3 + nc
            n_list[idx], n_list[target] = n_list[target], n_list[idx]
            neighbors.append(tuple(n_list))
    return neighbors

def count_inversions(state):
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv

def generate_solvable_state(goal):
    state_list = list(range(9))
    while True:
        random.shuffle(state_list)
        state = tuple(state_list)
        if count_inversions(state) % 2 == count_inversions(goal) % 2:
            return state

def manhattan_distance(state, goal):
    """ 启发式函数: 计算所有格子到目标位置的曼哈顿距离之和 """
    distance = 0
    for i in range(9):
        value = state[i]
        if value != 0:
            target_idx = goal.index(value)
            distance += abs(i // 3 - target_idx // 3) + abs(i % 3 - target_idx % 3)
    return distance

def a_star(start, goal):
    """ A* 搜索算法 """
    # 优先队列元素格式: (f_score, g_score, current_state, path)
    # f = g + h
    open_set = [(manhattan_distance(start, goal), 0, start, [start])]
    visited = {start: 0}
    
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        
        if current == goal:
            return path
            
        for neighbor in get_neighbors(current):
            new_g = g + 1
            if neighbor not in visited or new_g < visited[neighbor]:
                visited[neighbor] = new_g
                h_score = manhattan_distance(neighbor, goal)
                heapq.heappush(open_set, (new_g + h_score, new_g, neighbor, path + [neighbor]))
    return None

def print_puzzle(state):
    for i in range(0, 9, 3):
        print(f"{state[i]} {state[i+1]} {state[i+2]}")


if __name__ == "__main__":
    start_state = generate_solvable_state(GOAL_STATE)
    print(f"Random Start State (Seed: {STUDENT_ID}): {start_state}")
    print("Running Task B2: A* Search...")
    
    result = a_star(start_state, GOAL_STATE)
    
    if result:
        print(f"Total Moves: {len(result) - 1}")
        for step in result:
            print_puzzle(step)
            print("---")
    else:
        print("No solution found.")
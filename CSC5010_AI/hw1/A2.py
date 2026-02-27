# CSC5010 AI Homework 1
# A2: 8-Puzzle Problem (DFS)
# Xie Qianyu
# 225040249@link.cuhk.edu.cn

import random


STUDENT_ID = 225040249
random.seed(STUDENT_ID)
GOAL_STATE = (1, 2, 3, 8, 0, 4, 7, 6, 5)

def get_neighbors(state):
    neighbors = []
    idx = state.index(0)
    r, c = idx // 3, idx % 3
    # 移动顺序: Up, Down, Left, Right
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
        # 8数码问题可达性条件：两个状态的逆序对奇偶性必须相同
        if count_inversions(state) % 2 == count_inversions(goal) % 2:
            return state

def dfs(start, goal, limit=25):
    """ 深度优先搜索，带深度限制以防无限递归 """
    stack = [(start, [start])]
    visited = {start: 0}
    
    while stack:
        (state, path) = stack.pop()
        if state == goal:
            return path
        
        if len(path) <= limit:
            for neighbor in get_neighbors(state):
                # 只有当没访问过，或者当前路径比之前找到的更短时才探索
                if neighbor not in visited or visited[neighbor] > len(path):
                    visited[neighbor] = len(path)
                    stack.append((neighbor, path + [neighbor]))
    return None

def print_puzzle(state):
    for i in range(0, 9, 3):
        print(f"{state[i]} {state[i+1]} {state[i+2]}")


if __name__ == "__main__":
    start_state = generate_solvable_state(GOAL_STATE)
    print(f"Random Start State (Seed: {STUDENT_ID}): {start_state}")
    print("Running Task A2: DFS...")
    
    result = dfs(start_state, GOAL_STATE)
    
    if result:
        print(f"Total Moves: {len(result) - 1}")
        for step in result:
            print_puzzle(step)
            print("---")
    else:
        print("No solution found within depth limit.")
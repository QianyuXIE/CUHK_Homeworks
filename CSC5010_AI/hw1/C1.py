# CSC5010 AI Homework 1
# C1: Parity Test (DFS)
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

def dfs_check(start, goal, limit=25):
    """
    使用 DFS 尝试搜索。由于奇偶性不一致，目标是不可达的。
    设置深度限制 (limit) 防止在不可达的状态空间中无限搜索。
    """
    stack = [(start, [start])]
    visited = {start: 0}
    
    while stack:
        (state, path) = stack.pop()
        if state == goal:
            return path
        
        # 如果当前路径长度小于限制，继续向下搜索
        if len(path) <= limit:
            for neighbor in get_neighbors(state):
                if neighbor not in visited or visited[neighbor] > len(path):
                    visited[neighbor] = len(path)
                    stack.append((neighbor, path + [neighbor]))
    return None

# C1: 交换 2 和 8
# START_A1 = (2, 8, 3, 1, 6, 4, 7, 0, 5)
START_C1 = (8, 2, 3, 1, 6, 4, 7, 0, 5)
GOAL = (1, 2, 3, 8, 0, 4, 7, 6, 5)

print("--- Running Task C1: Parity Test (DFS) ---")
print(f"Testing start state with swapped 2 and 8: {START_C1}")

# 运行搜索
result = dfs_check(START_C1, GOAL)

if result:
    print(f"Success! Moves: {len(result)-1}")
else:
    print("\nResult: Goal state NOT reachable.")
    print("Explanation: Swapping 2 and 8 changes the parity of the state.")
    print("In the 8-puzzle problem, states with different parities belong to disconnected sets.")
    print("Therefore, the goal state cannot be reached from this modified start state.")
from collections import deque

# Goal state
GOAL_STATE = "123456780"

# Moves (up, down, left, right)
MOVES = {
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 5],
    3: [0, 4, 6],
    4: [1, 3, 5, 7],
    5: [2, 4, 8],
    6: [3, 7],
    7: [4, 6, 8],
    8: [5, 7]
}

def bfs_solve(start_state):
    """Solve the 8 Puzzle using BFS."""
    queue = deque([(start_state, start_state.index('0'), [])])
    visited = set()

    while queue:
        state, zero_pos, path = queue.popleft()
        if state == GOAL_STATE:
            return path + [state]

        if state in visited:
            continue
        visited.add(state)

        for move in MOVES[zero_pos]:
            new_state = list(state)
            new_state[zero_pos], new_state[move] = new_state[move], new_state[zero_pos]
            queue.append((''.join(new_state), move, path + [state]))

    return None

# Input: Initial state as a string
initial_state = "123406758"  # Example input

# Solve the puzzle
solution = bfs_solve(initial_state)

# Print the solution
if solution:
    print(f"Solution found in {len(solution) - 1} moves:")
    for step in solution:
        print(step[:3])
        print(step[3:6])
        print(step[6:])
        print()
else:
    print("The puzzle is unsolvable.")

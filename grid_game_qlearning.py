import numpy as np
import random

grid_size = 5
goal = (4, 4)
obstacles = [(2, 2), (3, 1)]

Q = np.zeros((grid_size, grid_size, 4))
actions = ["up", "down", "left", "right"]

alpha = 0.1
gamma = 0.9
epsilon = 0.2

def move(state, action):
    x, y = state
    if action == 0: x = max(0, x - 1)
    elif action == 1: x = min(grid_size - 1, x + 1)
    elif action == 2: y = max(0, y - 1)
    elif action == 3: y = min(grid_size - 1, y + 1)
    return (x, y)

def reward(state):
    if state == goal:
        return 10
    elif state in obstacles:
        return -5
    else:
        return -1

for episode in range(500):
    state = (0, 0)
    while state != goal:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state[0], state[1]])

        next_state = move(state, action)
        r = reward(next_state)

        Q[state[0], state[1], action] += alpha * (
            r + gamma * np.max(Q[next_state[0], next_state[1]])
            - Q[state[0], state[1], action]
        )

        state = next_state

print("Training complete!")
import random

def hill_climbing(f, x_start, step_size=0.1, max_iter=1000):
    current_x = x_start
    current_f = f(current_x)
    for _ in range(max_iter):
        neighbor_x = current_x + random.uniform(-step_size, step_size)
        neighbor_f = f(neighbor_x)
        if neighbor_f > current_f:
            current_x, current_f = neighbor_x, neighbor_f
    return current_x, current_f

objective_function = lambda x: -x**2 + 4*x
x_start = random.uniform(0, 5)
best_x, best_f = hill_climbing(objective_function, x_start)
print(f"Best solution: x = {best_x:.4f}, f(x) = {best_f:.4f}")

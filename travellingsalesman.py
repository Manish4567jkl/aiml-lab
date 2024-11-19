import itertools

# Distance matrix
distances = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Number of cities
num_cities = len(distances)

# All cities except the starting one (assume 0 as the starting city)
cities = range(1, num_cities)

# Initialize minimum path
min_path_cost = float('inf')
best_path = []

# Try all permutations of the cities
for perm in itertools.permutations(cities):
    # Add the starting city (0) to the beginning and end of the path
    path = (0,) + perm + (0,)
    # Calculate the total cost of the path
    cost = sum(distances[path[i]][path[i+1]] for i in range(num_cities))
    # Update minimum cost and best path
    if cost < min_path_cost:
        min_path_cost = cost
        best_path = path

# Output the result
print("Shortest Path:", best_path)
print("Minimum Cost:", min_path_cost)

import json
import random
import math

def euc_2d_distance(coord1, coord2):
    return round(math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2) + 0.5)

def nearest_neighbor_tsp(coords):
    n = len(coords)
    visited = [False] * n
    tour = []

    current_city = random.randint(0, n - 1)
    tour.append(current_city)
    visited[current_city] = True

    for _ in range(n - 1):
        next_city = min((i for i in range(n) if not visited[i]), key=lambda i: euc_2d_distance(coords[current_city], coords[i]))
        tour.append(next_city)
        visited[next_city] = True
        current_city = next_city

    return tour

def two_opt_swap(tour, i, j):
    new_tour = tour[:i]
    new_tour.extend(reversed(tour[i:j+1]))
    new_tour.extend(tour[j+1:])
    return new_tour

def two_opt_heuristic(coords, max_iter=1000):
    n = len(coords)
    tour = nearest_neighbor_tsp(coords)

    for _ in range(max_iter):
        best_tour = tour
        best_dist = sum(euc_2d_distance(coords[best_tour[i]], coords[best_tour[i+1]]) for i in range(n-1)) + euc_2d_distance(coords[best_tour[n-1]], coords[best_tour[0]])

        for i in range(1, n - 2):
            for j in range(i + 2, min(i + n - 2, n)):
                new_tour = two_opt_swap(tour, i, j)
                new_dist = sum(euc_2d_distance(coords[new_tour[i]], coords[new_tour[i+1]]) for i in range(n-1)) + euc_2d_distance(coords[new_tour[n-1]], coords[new_tour[0]])

                if new_dist < best_dist:
                    best_tour = new_tour
                    best_dist = new_dist

        if best_tour == tour:
            break

        tour = best_tour

    return tour

def is_valid_permutation(tour, n):
    return sorted(tour) == list(range(n))

def main():
    import sys
    input_json = sys.stdin.read()
    data = json.loads(input_json)
    requirements = data['requirements']
    catalog = data['catalog']

    random_seed = requirements.get('random_seed')
    if random_seed is not None:
        random.seed(random_seed)

    coords = [(city['x'], city['y']) for city in catalog['cities']]
    tour = two_opt_heuristic(coords)

    n = len(coords)
    assert is_valid_permutation(tour, n), "The tour is not a valid permutation of city IDs."

    result = {
        "selection": {
            "tour": tour
        }
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

import sys
import json
import math
import random
import time

def read_input():
    return json.load(sys.stdin)

def calculate_distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.floor(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + 0.5)

def nearest_neighbor_tour(cities):
    n = len(cities)
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        current = tour[-1]
        next_city = min(unvisited, key=lambda city: calculate_distance(cities[current], cities[city]))
        tour.append(next_city)
        unvisited.remove(next_city)
    return tour

def two_opt(tour, cities, max_time):
    n = len(cities)
    best_tour = tour[:]
    best_distance = sum(calculate_distance(cities[tour[i]], cities[tour[(i + 1) % n]]) for i in range(n))
    start_time = time.time()

    while time.time() - start_time < max_time:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 2, n):
                if j - i == 1:
                    continue
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_distance = sum(calculate_distance(cities[new_tour[i]], cities[new_tour[(i + 1) % n]]) for i in range(n))
                if new_distance < best_distance:
                    best_tour = new_tour[:]
                    best_distance = new_distance
                    improved = True
        if not improved:
            break
        tour = best_tour[:]

    return best_tour

def validate_tour(tour, n):
    return len(tour) == n and len(set(tour)) == n

def main():
    input_data = read_input()
    requirements = input_data["requirements"]
    catalog = input_data["catalog"]

    random.seed(requirements.get("random_seed", 0))
    time_limit_sec = requirements["time_limit_sec"]

    cities = [(city["x"], city["y"]) for city in catalog["cities"]]
    n = len(cities)

    initial_tour = nearest_neighbor_tour(cities)
    optimized_tour = two_opt(initial_tour, cities, time_limit_sec)

    if not validate_tour(optimized_tour, n):
        raise ValueError("Invalid tour")

    result = {
        "selection": {
            "tour": optimized_tour
        }
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()

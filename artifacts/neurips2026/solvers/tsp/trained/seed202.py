import sys
import random
import math
import json

def euc_2d(coord1, coord2):
    distance = math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
    return round(distance + 0.5)

def tour_length(cities, path):
    total_length = 0
    n = len(path)
    for i in range(n):
        total_length += euc_2d(cities[path[i]], cities[path[(i + 1) % n]])
    return total_length

def nearest_nearest_tsp(cities):
    n = len(cities)
    unvisited = set(range(n))
    visited = []

    current_city = random.choice(list(unvisited))
    unvisited.remove(current_city)
    visited.append(current_city)

    while unvisited:
        nearest_city = None
        min_distance = float('inf')
        for city in unvisited:
            distance = euc_2d(cities[current_city], cities[city])
            if distance < min_distance:
                nearest_city = city
                min_distance = distance

        unvisited.remove(nearest_city)
        visited.append(nearest_city)
        current_city = nearest_city

    return visited

def is_valid_tour(tour, n_cities):
    return len(tour) == n_cities and len(set(tour)) == n_cities

def main():
    import sys
    input_data = sys.stdin.read()
    data = json.loads(input_data)

    cities = [(item['x'], item['y']) for item in data['catalog']['cities']]

    n_cities = len(cities)
    tour = nearest_nearest_tsp(cities)

    if not is_valid_tour(tour, n_cities):
        print("The tour does not contain every city ID exactly once.", file=sys.stderr)
        sys.exit(1)

    result = json.dumps({'selection': {'tour': tour}})
    print(result)

if __name__ == '__main__':
    main()

import sys
import json
import math

def euclidean_distance(city1, city2):
    dx = city1[0] - city2[0]
    dy = city1[1] - city2[1]
    return math.floor(math.sqrt(dx * dx + dy * dy) + 0.5)

def nearest_neighbour_tour(cities):
    best_tour = None
    best_length = float('inf')
    n = len(cities)
    for start in range(n):
        visited = [False] * n
        visited[start] = True
        tour = [start]
        for _ in range(n - 1):
            min_dist = float('inf')
            next_city = -1
            for j in range(n):
                if not visited[j]:
                    dist = euclidean_distance(cities[tour[-1]], cities[j])
                    if dist < min_dist:
                        min_dist = dist
                        next_city = j
            visited[next_city] = True
            tour.append(next_city)
        tour_length = sum(euclidean_distance(cities[tour[i]], cities[tour[(i + 1) % n]]) for i in range(n))
        if tour_length < best_length:
            best_length = tour_length
            best_tour = tour
    return best_tour

def two_opt_swap(tour, i, k):
    new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
    return new_tour

def two_opt_heuristic(tour, cities):
    best_tour = tour
    best_length = sum(euclidean_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))
    n = len(tour)
    while True:
        improved = False
        for i in range(n):
            for k in range(i, n):
                new_tour = two_opt_swap(tour, i, k)
                new_length = sum(euclidean_distance(cities[new_tour[i]], cities[new_tour[(i + 1) % n]]) for i in range(n))
                if new_length < best_length:
                    best_length = new_length
                    best_tour = new_tour
                    improved = True
        if not improved:
            break
        tour = best_tour
    return best_tour

def main():
    import sys
    import json

    data = sys.stdin.read()
    info = json.loads(data)
    cities = [(city['x'], city['y']) for city in info['catalog']['cities']]

    initial_tour = nearest_neighbour_tour(cities)
    best_tour = two_opt_heuristic(initial_tour, cities)

    print(json.dumps({"selection": {"tour": best_tour}}))

if __name__ == "__main__":
    main()

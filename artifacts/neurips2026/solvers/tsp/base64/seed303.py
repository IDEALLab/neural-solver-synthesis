import sys
import json
import math
import random
import time

def euc_2d_distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.floor(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) + 0.5)

def calculate_tour_length(tour, cities):
    n = len(tour)
    total_distance = 0
    for i in range(n):
        total_distance += euc_2d_distance(cities[tour[i]], cities[tour[(i + 1) % n]])
    return total_distance

def generate_random_tour(n):
    return random.sample(range(n), n)

def tournament_selection(population, fitness, tournament_size=3):
    selected = random.sample(range(len(population)), tournament_size)
    best_index = min(selected, key=lambda i: fitness[i])
    return population[best_index]

def order_crossover(parent1, parent2):
    n = len(parent1)
    start, end = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[start:end+1] = parent1[start:end+1]
    for i in range(n):
        if parent2[i] not in child:
            for j in range(n):
                if child[j] == -1:
                    child[j] = parent2[i]
                    break
    return child

def swap_mutation(tour, mutation_rate=0.01):
    n = len(tour)
    for i in range(n):
        if random.random() < mutation_rate:
            j = random.randint(0, n-1)
            tour[i], tour[j] = tour[j], tour[i]
    return tour

def two_opt(tour, cities):
    n = len(tour)
    best_tour = tour[:]
    best_distance = calculate_tour_length(tour, cities)
    improved = True
    while improved:
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n):
                new_tour = tour[:]
                new_tour[i:j+1] = reversed(new_tour[i:j+1])
                new_distance = calculate_tour_length(new_tour, cities)
                if new_distance < best_distance:
                    best_tour = new_tour[:]
                    best_distance = new_distance
                    improved = True
    return best_tour

def genetic_algorithm(cities, time_limit_sec, population_size=50, generations=1000):
    n = len(cities)
    population = [generate_random_tour(n) for _ in range(population_size)]
    fitness = [calculate_tour_length(tour, cities) for tour in population]

    start_time = time.time()
    best_tour = population[fitness.index(min(fitness))]

    for gen in range(generations):
        if time.time() - start_time > time_limit_sec:
            break

        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            child = order_crossover(parent1, parent2)
            child = swap_mutation(child)
            new_population.append(child)

        population = new_population
        fitness = [calculate_tour_length(tour, cities) for tour in population]

        if min(fitness) < calculate_tour_length(best_tour, cities):
            best_tour = population[fitness.index(min(fitness))]

    best_tour = two_opt(best_tour, cities)
    return best_tour

def main():
    import sys
    import json

    input_data = sys.stdin.read()
    data = json.loads(input_data)

    requirements = data["requirements"]
    catalog = data["catalog"]

    random.seed(requirements.get("random_seed", 42))
    time_limit_sec = requirements["time_limit_sec"]

    cities = [(city["x"], city["y"]) for city in catalog["cities"]]

    best_tour = genetic_algorithm(cities, time_limit_sec)

    result = {
        "selection": {
            "tour": best_tour
        }
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()

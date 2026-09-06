import sys
import json
import math
import random
import time

def read_input():
    input_data = sys.stdin.read()
    data = json.loads(input_data)
    return data

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
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]

def two_opt(tour, cities):
    n = len(tour)
    best_tour = tour[:]
    best_distance = calculate_tour_length(tour, cities)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 2, n):
                new_tour = tour[:]
                new_tour[i:j+1] = reversed(new_tour[i:j+1])
                new_distance = calculate_tour_length(new_tour, cities)
                if new_distance < best_distance:
                    best_tour = new_tour[:]
                    best_distance = new_distance
                    improved = True
    return best_tour

def genetic_algorithm(cities, population_size=100, generations=1000, mutation_rate=0.01):
    n = len(cities)
    population = [generate_random_tour(n) for _ in range(population_size)]
    fitness = [calculate_tour_length(tour, cities) for tour in population]

    for generation in range(generations):
        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            child = order_crossover(parent1, parent2)
            swap_mutation(child, mutation_rate)
            new_population.append(child)
        population = new_population
        fitness = [calculate_tour_length(tour, cities) for tour in population]

    best_tour = min(population, key=lambda tour: calculate_tour_length(tour, cities))
    return best_tour

def main():
    data = read_input()
    requirements = data["requirements"]
    catalog = data["catalog"]
    cities = [(city["x"], city["y"]) for city in catalog["cities"]]
    random.seed(requirements["random_seed"])
    start_time = time.time()
    tour = genetic_algorithm(cities, generations=min(1000, int(requirements["time_limit_sec"] * 100)))
    end_time = time.time()
    print(json.dumps({"selection": {"tour": tour}}))

if __name__ == "__main__":
    main()

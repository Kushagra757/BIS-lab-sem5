import random

# Configuration parameters
TARGET = "HELLO WORLD"
POPULATION_SIZE = 100
MUTATION_RATE = 0.01  # 1% chance for each character to mutate
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

def generate_individual(length):
    """Creates a random string of the specified length from the charset."""
    return "".join(random.choice(CHARSET) for _ in range(length))

def calculate_fitness(individual, target):
    """Calculates how many characters match the target string exactly."""
    return sum(1 for ind_char, tar_char in zip(individual, target) if ind_char == tar_char)

def crossover(parent1, parent2):
    """Combines two parents using a single-point crossover."""
    split_point = random.randint(0, len(parent1) - 1)
    child = parent1[:split_point] + parent2[split_point:]
    return child

def mutate(individual, mutation_rate, charset):
    """Randomly mutates genes (characters) based on the mutation rate."""
    individual_list = list(individual)
    for i in range(len(individual_list)):
        if random.random() < mutation_rate:
            individual_list[i] = random.choice(charset)
    return "".join(individual_list)

def main():
    # 1. Initialize the population
    population = [generate_individual(len(TARGET)) for _ in range(POPULATION_SIZE)]
    generation = 0

    while True:
        generation += 1

        # 2. Evaluate fitness for the current population
        # We store tuples of (fitness_score, individual_string)
        fitness_scores = [(calculate_fitness(ind, TARGET), ind) for ind in population]

        # Sort population by fitness in descending order
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        best_fitness, best_individual = fitness_scores[0]

        print(f"Gen {generation} | Best Individual: '{best_individual}' | Fitness: {best_fitness}/{len(TARGET)}")

        # Check if the optimization target is reached
        if best_individual == TARGET:
            print(f"\nTarget reached in generation {generation}!")
            break

        # 3. Selection (Elitism: carry the top 10% directly to the next generation)
        next_generation = [ind for score, ind in fitness_scores[:int(POPULATION_SIZE * 0.10)]]

        # Mating pool consists of the top 50% performers
        mating_pool = [ind for score, ind in fitness_scores[:int(POPULATION_SIZE * 0.50)]]

        # 4. Reproduction (Crossover and Mutation)
        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(mating_pool)
            parent2 = random.choice(mating_pool)

            child = crossover(parent1, parent2)
            child = mutate(child, MUTATION_RATE, CHARSET)

            next_generation.append(child)

        population = next_generation

if __name__ == "__main__":
    main()

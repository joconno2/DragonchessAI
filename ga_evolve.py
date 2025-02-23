# ga_evolve.py
import random
import numpy as np
import time
import csv
from concurrent.futures import ProcessPoolExecutor
from bots.ga_bot import GeneticBot, decode_chromosome
from game import Game
from ai import RandomAI

# GA parameters.
POP_SIZE = 20
CHROMOSOME_LENGTH = 14 * 8  # 14 genes × 8 bits = 112 bits.
GENERATIONS = 50
MUTATION_RATE = 0.01  # Mutation probability per bit.
EVAL_GAMES = 3       # Each individual plays 3 games against a RandomAI.
# Bonus factor used to reward quicker wins.
BONUS_FACTOR = 300.0

def random_chromosome():
    """Generate a random binary string of length CHROMOSOME_LENGTH."""
    return ''.join(random.choice(['0', '1']) for _ in range(CHROMOSOME_LENGTH))

def roulette_wheel_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choice(population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for chrom, fit in zip(population, fitnesses):
        current += fit
        if current >= pick:
            return chrom
    return population[-1]

def single_point_crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome, mutation_rate=MUTATION_RATE):
    chrom_list = list(chromosome)
    for i in range(len(chrom_list)):
        if random.random() < mutation_rate:
            chrom_list[i] = '1' if chrom_list[i] == '0' else '0'
    return ''.join(chrom_list)

def evaluate_individual(chromosome, num_games=EVAL_GAMES):
    """
    Evaluate fitness by having the GeneticBot (playing as Gold) compete
    in num_games against a RandomAI opponent (Scarlet). For each game,
    if GeneticBot wins the score is 1 + BONUS_FACTOR/move_count (i.e. winning faster yields a higher score),
    a draw is given 0.5 and a loss 0.0. The total fitness is the sum over games.
    """
    total_score = 0.0
    for _ in range(num_games):
        game = Game()
        ga_bot = GeneticBot(game, "Gold", chromosome)
        opponent = RandomAI(game, "Scarlet")
        moves = 0
        while not game.game_over:
            if game.current_turn == "Gold":
                move = ga_bot.choose_move()
            else:
                move = opponent.choose_move()
            if move:
                game.make_move(move)
                moves += 1
            game.update()
        if game.winner == "Gold":
            game_score = 1.0 + (BONUS_FACTOR / moves)
        elif game.winner == "Draw":
            game_score = 0.5
        else:
            game_score = 0.0
        total_score += game_score
    return total_score

def roulette_wheel_selection_pair(population, fitnesses):
    """Helper: select two parents independently using roulette wheel."""
    return (roulette_wheel_selection(population, fitnesses),
            roulette_wheel_selection(population, fitnesses))

def main():
    # Open CSV log file to record every individual's chromosome and fitness per generation.
    with open("ga_log.csv", "w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Generation", "Chromosome", "Fitness"])
        
        # Initialize population.
        population = [random_chromosome() for _ in range(POP_SIZE)]
        best_overall = None
        best_fitness_overall = -1

        for gen in range(GENERATIONS):
            print(f"Generation {gen}")
            # Evaluate the entire population concurrently.
            with ProcessPoolExecutor() as executor:
                fitnesses = list(executor.map(evaluate_individual, population))
            # Log each individual's chromosome and fitness.
            for chrom, fitness in zip(population, fitnesses):
                log_writer.writerow([gen, chrom, fitness])
            log_file.flush()

            best_idx = np.argmax(fitnesses)
            best_gen_fitness = fitnesses[best_idx]
            print(f" Best fitness in generation {gen}: {best_gen_fitness}/{EVAL_GAMES} game score")
            if best_gen_fitness > best_fitness_overall:
                best_fitness_overall = best_gen_fitness
                best_overall = population[best_idx]
            # Generate new population.
            new_population = []
            while len(new_population) < POP_SIZE:
                parent1, parent2 = roulette_wheel_selection_pair(population, fitnesses)
                child1, child2 = single_point_crossover(parent1, parent2)
                child1 = mutate(child1)
                child2 = mutate(child2)
                new_population.append(child1)
                if len(new_population) < POP_SIZE:
                    new_population.append(child2)
            population = new_population

        print("Evolution complete.")
        print(f"Best overall fitness: {best_fitness_overall}/{EVAL_GAMES} game score")
        decoded_values = decode_chromosome(best_overall)
        print("Best chromosome piece values array:")
        print(decoded_values)

if __name__ == "__main__":
    main()

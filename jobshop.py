import random

#GLOBALS
DATASET_PATH = './data/la35.txt'
NUM_JOBS = 0
NUM_MACHINES = 0
JOBS = None
TURNAMNET_SIZE = 0


def load_instance(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()

    NUM_JOBS, NUM_MACHINES = map(int, lines[0].split())

    JOBS = []

    for line in lines[1:]:
        values = list(map(int, line.split()))

        operations = []

        for i in range(0, len(values), 2):
            machine = values[i]
            processing_time = values[i+1]

            operations.append((machine, processing_time))

        JOBS.append(operations)

    return NUM_JOBS, NUM_MACHINES, JOBS


# print("Jobs:", NUM_JOBS)
# print("Machines:", NUM_MACHINES)
# print("Job 0:", JOBS[0])

def create_chromosome():
    '''
    Creates a random chromosome for the job shop scheduling problem.
    Each job is represented by its ID, and the chromosome is a list of job IDs.
    Each job ID appears in the chromosome as many times as it has operations (equal to the number of machines).
    '''
    chromosome = []

    for job_id in range(NUM_JOBS):
        chromosome.extend([job_id] * NUM_MACHINES)

    random.shuffle(chromosome)

    return chromosome

# chromosome = create_chromosome()

# print("Chromosome:", chromosome)
# print(len(chromosome))


def decode_chromosome(chromosome):
    '''
    Decodes a chromosome into a schedule and calculates the makespan.
    The schedule is represented as a list of dictionaries, where each dictionary contains:
    - job: the job ID
    - operation: the operation ID (0 to NUM_MACHINES-1)
    - machine: the machine ID for this operation
    - start: the start time of the operation
    - finish: the finish time of the operation
    '''
    job_counters = [0] * NUM_JOBS

    job_ready_times = [0] * NUM_JOBS
    machine_ready_times = [0] * NUM_MACHINES

    schedule = []

    for job_id in chromosome:
        operation_id = job_counters[job_id]

        machine, processing_time = JOBS[job_id][operation_id]

        start_time = max(
            job_ready_times[job_id],
            machine_ready_times[machine]
        )
        finish_time = start_time + processing_time

        schedule.append({
            "job": job_id,
            "operation": operation_id,
            "machine": machine,
            "start": start_time,
            "finish": finish_time
        })

        job_ready_times[job_id] = finish_time
        machine_ready_times[machine] = finish_time

        job_counters[job_id] += 1

    makespan = max(machine_ready_times)

    return schedule, makespan


def fitness(chromosome):
    _, makespan = decode_chromosome(chromosome)

    return makespan

def tournament_selection(
    population,
    turnament_size = 2
):
    candidate_list = random.sample(
        population,
        turnament_size
    )

    candidate_list_sorted = sorted(candidate_list, key=lambda item: item[1])
    return candidate_list_sorted[0]

def crossover_pox(parent1, parent2):
    list_len = len(parent1)
    child1 = [None] * list_len
    child2 = [None] * list_len

    selected_JOBS = set(random.sample(range(NUM_JOBS),NUM_JOBS // 2))


    remaining_for_child1 = []
    for job in parent2:
        if job not in selected_JOBS:
            remaining_for_child1.append(job)

    remaining_for_child2 = []
    for job in parent1:
            if job not in selected_JOBS:
                remaining_for_child2.append(job)

    for index in range(list_len):
        if parent1[index] in selected_JOBS:
            child1[index] = parent1[index]
        if parent2[index] in selected_JOBS:
            child2[index] = parent2[index]

    p2_idx = 0
    p1_idx = 0
    for index in range(list_len):
        if child1[index] is None:
            child1[index] = remaining_for_child1[p2_idx]
            p2_idx += 1
        if child2[index] is None:
            child2[index] = remaining_for_child2[p1_idx]
            p1_idx += 1

    return child1, child2

def mutate(chromosome):
    '''
    Perform mutation on a chromosome by swapping two random genes.
    '''
    chromosome_copy = chromosome.copy()

    index1, index2 = random.sample(range(len(chromosome_copy)), 2)

    chromosome_copy[index1], chromosome_copy[index2] = chromosome_copy[index2], chromosome_copy[index1]

    return chromosome_copy

def create_population(population_size):
    '''
    Create an initial population of chromosomes for the job shop scheduling problem.
    Each chromosome is a random permutation of job IDs, where each job ID appears as many times as it has operations (equal to the number of machines).
    '''
    population = [
        create_chromosome()
        for _ in range(population_size)
    ]
    
    population_fitness_list = []
    for chromosome in population:
        fitness_value = fitness(chromosome)
        population_fitness_list.append([chromosome, fitness_value])

    return population_fitness_list


# =========== Main Genetic Algorithm Function ===========

def genetic_algorithm(
    population_size=100,
    generations=1000,
    mutation_rate=0.1,
    crossover_rate=0.8
):
    population_fitness_list = create_population(population_size)

    best_entry = min(population_fitness_list, key=lambda item: item[1])

    best_chromosome = best_entry[0].copy()
    best_fitness = best_entry[1]

    for generation in range(1,generations + 1):
        generation_best = min(population_fitness_list, key=lambda item: item[1])

        if generation_best[1] < best_fitness:
            best_chromosome = generation_best.copy()

        print(
            f"Generation {generation}: Best Fitness = {best_chromosome[1]}"
        )

        new_population = []

        while len(new_population) < population_size:
            parent1 = tournament_selection(
                population_fitness_list
            )
            parent2 = tournament_selection(
                population_fitness_list
            )

            parent_chromosom1 = parent1[0]
            parent_chromosom2 = parent2[0]

            # Crossover
            crossover_probability = random.random()

            if crossover_probability < crossover_rate:
                child_chromosome1, child_chromosome2 = crossover_pox(
                    parent_chromosom1,
                    parent_chromosom2
                )

            else:
                child_chromosome1 = parent_chromosom1.copy()
                child_chromosome2 = parent_chromosom2.copy()

            # Mutation
            mutation_probability1 = random.random()

            if mutation_probability1 < mutation_rate:
                child_chromosome1 = mutate(child_chromosome1)

            mutation_probability2 = random.random()

            if mutation_probability2 < mutation_rate:
                child_chromosome2 = mutate(child_chromosome2)

            child1 = [child_chromosome1,fitness(child_chromosome1)]
            child2 = [child_chromosome2,fitness(child_chromosome2)]

            # Add children to new population
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population_fitness_list = new_population

    final_best = min(
        population_fitness_list,
        key=lambda item: item[1]
    )
    if final_best[1] < best_chromosome[1]:
        best_chromosome = final_best.copy()

    return best_chromosome
    
if __name__ == "__main__":
    NUM_JOBS, NUM_MACHINES, JOBS = load_instance("./data/la01.txt")

    for _ in range(0,1,1):
        best_solution = genetic_algorithm(
            population_size=200,
            generations=1000,
            mutation_rate=0.1,
            crossover_rate=0.8
        )

        print("Best solution found:", best_solution[0])
        schedule, makespan = decode_chromosome(best_solution[0])
        print("\nSchedule:")
        for operation in schedule:
            print(operation)
        print("Makespan:", makespan)
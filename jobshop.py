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

def create_chromosome(NUM_JOBS, NUM_MACHINES):
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

chromosome = create_chromosome(NUM_JOBS, NUM_MACHINES)

# print("Chromosome:", chromosome)
# print(len(chromosome))


def decode_chromosome(chromosome, JOBS, NUM_JOBS, NUM_MACHINES):
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

# ========== Test the functions ==========
# chromosome = create_chromosome(
#     NUM_JOBS,
#     NUM_MACHINES
# )

# schedule, makespan = decode_chromosome(
#     chromosome,
#     JOBS,
#     NUM_JOBS,
#     NUM_MACHINES
# )

# print("Chromosome:")
# print(chromosome)

# print("\nSchedule:")
# for operation in schedule:
#     print(operation)

# print("\nMakespan:")
# print(makespan)

# =======================================

def fitness(chromosome, JOBS, NUM_JOBS, NUM_MACHINES):
    _, makespan = decode_chromosome(
        chromosome,
        JOBS,
        NUM_JOBS,
        NUM_MACHINES
    )

    return makespan

def tournament_selection(
    population,
    JOBS,
    NUM_JOBS,
    NUM_MACHINES,
    turnament_size = 2
):
    candidate_list = random.sample(
        population,
        turnament_size
    )

    candidate_fitness_list = []

    for candidate in candidate_list:
        candidate_fitness_list.append([fitness(candidate, JOBS, NUM_JOBS, NUM_MACHINES),candidate])

    candidate_fitness_list_sorted = sorted(candidate_fitness_list, key=lambda item: item[0])

    return candidate_fitness_list_sorted[0][1]

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

def create_population(population_size, NUM_JOBS, NUM_MACHINES):
    '''
    Create an initial population of chromosomes for the job shop scheduling problem.
    Each chromosome is a random permutation of job IDs, where each job ID appears as many times as it has operations (equal to the number of machines).
    '''
    population = [
        create_chromosome(NUM_JOBS, NUM_MACHINES)
        for _ in range(population_size)
    ]

    return population


# =========== Main Genetic Algorithm Function ===========

def genetic_algorithm(
    JOBS,
    NUM_JOBS,
    NUM_MACHINES,
    population_size=100,
    generations=1000,
    mutation_rate=0.1,
    crossover_rate=0.8
):
    population = [
        create_chromosome(NUM_JOBS, NUM_MACHINES)
        for _ in range(population_size)
    ]

    best_chromosome = min(
        population,
        key=lambda chromosome: fitness(
            chromosome,
            JOBS,
            NUM_JOBS,
            NUM_MACHINES
        )
    ).copy()

    for generation in range(
        1,
        generations + 1
    ):

        generation_best = min(
            population,
            key=lambda chromosome: fitness(
                chromosome,
                JOBS,
                NUM_JOBS,
                NUM_MACHINES
            )
        )

        if fitness(
            generation_best,
            JOBS,
            NUM_JOBS,
            NUM_MACHINES
        ) < fitness(
            best_chromosome,
            JOBS,
            NUM_JOBS,
            NUM_MACHINES
        ):
            best_chromosome = generation_best.copy()

        print(
            f"Generation {generation}: Best Fitness = {fitness(best_chromosome, JOBS, NUM_JOBS, NUM_MACHINES)}"
        )

        new_population = []

        while len(new_population) < population_size:
            parent1 = tournament_selection(
                population,
                JOBS,
                NUM_JOBS,
                NUM_MACHINES
            )
            parent2 = tournament_selection(
                population,
                JOBS,
                NUM_JOBS,
                NUM_MACHINES
            )

            # Crossover
            crossover_probability = random.random()

            if crossover_probability < crossover_rate:
                child1, child2 = crossover_pox(
                    parent1,
                    parent2
                )

            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutation
            mutation_probability1 = random.random()

            if mutation_probability1 < mutation_rate:
                child1 = mutate(child1)

            mutation_probability2 = random.random()

            if mutation_probability2 < mutation_rate:
                child2 = mutate(child2)

            # Add children to new population
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    final_best = min(
        population,
        key=lambda chromosome: fitness(
            chromosome,
            JOBS,
            NUM_JOBS,
            NUM_MACHINES
        )
    )
    if fitness(
        final_best,
        JOBS,
        NUM_JOBS,
        NUM_MACHINES
    ) < fitness(
        best_chromosome,
        JOBS,
        NUM_JOBS,
        NUM_MACHINES
    ):
        best_chromosome = final_best.copy()

    return best_chromosome
    
if __name__ == "__main__":
    NUM_JOBS, NUM_MACHINES, JOBS = load_instance("./data/la35.txt")

    for _ in range(0,1,1):
        best_solution = genetic_algorithm(
            JOBS,
            NUM_JOBS,
            NUM_MACHINES,
            population_size=200,
            generations=1000,
            mutation_rate=0.1,
            crossover_rate=0.8
        )

        print("Best solution found:", best_solution)
        schedule, makespan = decode_chromosome(
            best_solution,
            JOBS,
            NUM_JOBS,
            NUM_MACHINES
        )
        # print("\nSchedule:")
        # for operation in schedule:
        #     print(operation)
        print("Makespan:", makespan)
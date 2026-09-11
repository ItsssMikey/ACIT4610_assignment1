import random

from matplotlib.pylab import size

def load_instance(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()

    num_jobs, num_machines = map(int, lines[0].split())

    jobs = []

    for line in lines[1:]:
        values = list(map(int, line.split()))

        operations = []

        for i in range(0, len(values), 2):
            machine = values[i]
            processing_time = values[i+1]

            operations.append((machine, processing_time))

        jobs.append(operations)

    return num_jobs, num_machines, jobs

num_jobs, num_machines, jobs = load_instance("./data/la01.txt")

# print("Jobs:", num_jobs)
# print("Machines:", num_machines)
# print("Job 0:", jobs[0])

def create_chromosome(num_jobs, num_machines):
    '''
    Creates a random chromosome for the job shop scheduling problem.
    Each job is represented by its ID, and the chromosome is a list of job IDs.
    Each job ID appears in the chromosome as many times as it has operations (equal to the number of machines).
    '''
    chromosome = []

    for job_id in range(num_jobs):
        chromosome.extend([job_id] * num_machines)

    random.shuffle(chromosome)

    return chromosome

chromosome = create_chromosome(num_jobs, num_machines)

# print("Chromosome:", chromosome)
# print(len(chromosome))


def decode_chromosome(chromosome, jobs, num_jobs, num_machines):
    '''
    Decodes a chromosome into a schedule and calculates the makespan.
    The schedule is represented as a list of dictionaries, where each dictionary contains:
    - job: the job ID
    - operation: the operation ID (0 to num_machines-1)
    - machine: the machine ID for this operation
    - start: the start time of the operation
    - finish: the finish time of the operation
    '''
    job_counters = [0] * num_jobs

    job_ready_times = [0] * num_jobs
    machine_ready_times = [0] * num_machines

    schedule = []

    for job_id in chromosome:
        operation_id = job_counters[job_id]

        machine, processing_time = jobs[job_id][operation_id]

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
#     num_jobs,
#     num_machines
# )

# schedule, makespan = decode_chromosome(
#     chromosome,
#     jobs,
#     num_jobs,
#     num_machines
# )

# print("Chromosome:")
# print(chromosome)

# print("\nSchedule:")
# for operation in schedule:
#     print(operation)

# print("\nMakespan:")
# print(makespan)

# =======================================

def fitness(chromosome, jobs, num_jobs, num_machines):
    _, makespan = decode_chromosome(
        chromosome,
        jobs,
        num_jobs,
        num_machines
    )

    return makespan

def tournament_selection(
    population,
    jobs,
    num_jobs,
    num_machines
):
    candidate1, candidate2 = random.sample(
        population,
        2
    )

    if fitness(
        candidate1,
        jobs,
        num_jobs,
        num_machines
    ) <= fitness(
        candidate2,
        jobs,
        num_jobs,
        num_machines
    ):
        return candidate1
    return candidate2

def crossover(parent1, parent2):
    size = len(parent1)
    child1 = [None] * size
    child2 = [None] * size

    gene1 = random.choice(parent1)
    gene2 = random.choice(parent2)

    remaining_for_child1 = parent2.copy()
    remaining_for_child2 = parent1.copy()

    for index in range(size):
        if parent1[index] == gene1:
            child1[index] = gene1
            remaining_for_child1.remove(gene1)

        if parent2[index] == gene2:
            child2[index] = gene2
            remaining_for_child2.remove(gene2)

    for index in range(size):
        if child1[index] is None:
            child1[index] = remaining_for_child1.pop(0)

        if child2[index] is None:
            child2[index] = remaining_for_child2.pop(0)

    return child1, child2

def mutate(chromosome):
    '''
    Perform mutation on a chromosome by swapping two random genes.
    '''
    chromosome_copy = chromosome.copy()

    index1, index2 = random.sample(range(len(chromosome_copy)), 2)

    chromosome_copy[index1], chromosome_copy[index2] = chromosome_copy[index2], chromosome_copy[index1]

    return chromosome_copy

def create_population(population_size, num_jobs, num_machines):
    '''
    Create an initial population of chromosomes for the job shop scheduling problem.
    Each chromosome is a random permutation of job IDs, where each job ID appears as many times as it has operations (equal to the number of machines).
    '''
    population = [
        create_chromosome(num_jobs, num_machines)
        for _ in range(population_size)
    ]

    return population


# =========== Main Genetic Algorithm Function ===========

def genetic_algorithm(
    jobs,
    num_jobs,
    num_machines,
    population_size=100,
    generations=1000,
    mutation_rate=0.1,
    crossover_rate=0.8
):
    population = [
        create_chromosome(num_jobs, num_machines)
        for _ in range(population_size)
    ]

    best_chromosome = min(
        population,
        key=lambda chromosome: fitness(
            chromosome,
            jobs,
            num_jobs,
            num_machines
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
                jobs,
                num_jobs,
                num_machines
            )
        )

        if fitness(
            generation_best,
            jobs,
            num_jobs,
            num_machines
        ) < fitness(
            best_chromosome,
            jobs,
            num_jobs,
            num_machines
        ):
            best_chromosome = generation_best.copy()

        print(
            f"Generation {generation}: Best Fitness = {fitness(best_chromosome, jobs, num_jobs, num_machines)}"
        )

        new_population = []

        while len(new_population) < population_size:
            parent1 = tournament_selection(
                population,
                jobs,
                num_jobs,
                num_machines
            )
            parent2 = tournament_selection(
                population,
                jobs,
                num_jobs,
                num_machines
            )

            # Crossover
            crossover_probability = random.random()

            if crossover_probability < crossover_rate:
                child1, child2 = crossover(
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
            jobs,
            num_jobs,
            num_machines
        )
    )
    if fitness(
        final_best,
        jobs,
        num_jobs,
        num_machines
    ) < fitness(
        best_chromosome,
        jobs,
        num_jobs,
        num_machines
    ):
        best_chromosome = final_best.copy()

    return best_chromosome
    
if __name__ == "__main__":
    best_solution = genetic_algorithm(
        jobs,
        num_jobs,
        num_machines,
        population_size=100,
        generations=2000,
        mutation_rate=0.1,
        crossover_rate=0.8
    )

    print("Best solution found:", best_solution)
    schedule, makespan = decode_chromosome(
        best_solution,
        jobs,
        num_jobs,
        num_machines
    )
    print("\nSchedule:")
    for operation in schedule:
        print(operation)
    print("Makespan:", makespan)
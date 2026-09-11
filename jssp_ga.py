import json
import random

FILENAME = "la01.txt"

def read_file(filename: str):
    with open(f"./data/{filename}", 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]

    header = lines[0].split()
    num_jobs = int(header[0])
    num_machines = int(header[1])

    machine_matrix = []
    duration_matrix = []

    for line in lines[1 : num_jobs + 1]:
        values = [int(val) for val in line.split()]

        job_machines = values[0::2]
        job_durations = values[1::2]

        machine_matrix.append(job_machines)
        duration_matrix.append(job_durations)

    return num_jobs, num_machines, machine_matrix, duration_matrix

def create_chromosome(num_jobs: int, num_machines: int):
    chromosome = list(range(num_jobs)) * num_machines
    random.shuffle(chromosome)
    return chromosome

def initialize_population(population_size: int, num_jobs: int, num_machines: int):
    initial_population = []
    for _ in range(0,population_size,1):
        initial_population.append(create_chromosome(num_jobs,num_machines))
    return initial_population

def find_semi_active_start_time(job_ready_time, machine_ready_time):
    return max(job_ready_time, machine_ready_time)

def decode_chromosome(chromosome: list, num_jobs: int, num_machines: int, machine_matrix: list, duration_matrix:list):
    job_step_counter   = [0]*num_jobs
    job_ready_time     = [0]*num_jobs
    machine_ready_time = [0]*num_machines
    machine_timelines  = [[]for _ in range(num_machines)]
    all_operations     = []
    c_max              = 0

    for job_id in chromosome:
        current_step = job_step_counter[job_id]
        target_machine = machine_matrix[job_id][current_step]
        duration       = duration_matrix[job_id][current_step]

        start_time = find_semi_active_start_time(job_ready_time[job_id], machine_ready_time[target_machine])
        end_time   = start_time + duration

        job_ready_time[job_id]             = end_time
        machine_ready_time[target_machine] = end_time
        job_step_counter[job_id]           = current_step + 1

        operation_entry = {
            "job_id": job_id,
            "step": current_step,
            "machine_id": target_machine,
            "start_time": start_time,
            "end_time": end_time,
        }

        machine_timelines[target_machine].append(operation_entry)
        all_operations.append(operation_entry)

        if end_time > c_max:
            c_max = end_time

    return {
        "makespan": c_max,
        "operations": all_operations,
        "machine_timelines": machine_timelines,
    }

def evaluate_fitness(chromosome: list, num_jobs: int, num_machines: int, machine_matrix: list, duration_matrix:list):
    job_step_counter   = [0]*num_jobs
    job_ready_time     = [0]*num_jobs
    machine_ready_time = [0]*num_machines
    c_max              = 0

    for job_id in chromosome:
            current_step = job_step_counter[job_id]
            target_machine = machine_matrix[job_id][current_step]
            duration       = duration_matrix[job_id][current_step]

            start_time = max(job_ready_time[job_id], machine_ready_time[target_machine])
            end_time   = start_time + duration

            job_ready_time[job_id]             = end_time
            machine_ready_time[target_machine] = end_time
            job_step_counter[job_id]           = current_step + 1

            if end_time > c_max:
                c_max = end_time
    return c_max

def tournament_selection(population: list, tournament_size: int, num_jobs: int, num_machines: int, machine_matrix: list, duration_matrix:list):
    candidate_list = random.sample(
        population,
        tournament_size,
    )
    fitness_list = []

    for candidate in candidate_list:
        fitness_list.append([evaluate_fitness(candidate,num_jobs,num_machines,machine_matrix,duration_matrix),candidate])

    fitness_list = sorted(fitness_list, key=lambda item: item[0])
    return fitness_list[0][1]

def crossover (parent1: list, parent2: list, num_jobs: int, num_machines: int, crossover_probability: float):
    random_prob = random.random()
    if random_prob >= crossover_probability:
        return parent1.copy(), parent2.copy()

    list_len = len(parent1)

    child1 = [None] * list_len
    child2 = [None] * list_len

    target_indices = random.sample(range(list_len), list_len // 2)

    for index in target_indices:
        child1[index] = parent1[index]
    indices_child2 = [i for i, val in enumerate(child1) if val is None]
    for index in indices_child2:
        child2[index] = parent2[index]

    counts_child1 = [0] * num_jobs
    for job in child1:
        if job is not None:
            counts_child1[job] += 1

    p2_index = 0
    for i in range(list_len):
        if child1[i] is None:
            while p2_index < list_len:
                candidate_job = parent2[p2_index]
                p2_index += 1
                if counts_child1[candidate_job] < num_machines:
                    child1[i] = candidate_job
                    counts_child1[candidate_job] += 1
                    break

    counts_child2 = [0] * num_jobs
    for job in child2:
        if job is not None:
            counts_child2[job] += 1

    p1_index = 0
    for i in range(list_len):
        if child2[i] is None:
            while p1_index < list_len:
                candidate_job = parent1[p1_index]
                p1_index += 1
                if counts_child2[candidate_job] < num_machines:
                    child2[i] = candidate_job
                    counts_child2[candidate_job] += 1
                    break

    return child1, child2

def main():
    population_size = 10
    num_jobs, num_machines, machine_matrix, duration_matrix = read_file(FILENAME)
    initial_population = initialize_population(population_size, num_jobs, num_machines)
    gantt_chart_data = decode_chromosome(initial_population[0],num_jobs,num_machines,machine_matrix,duration_matrix)
    fitness = evaluate_fitness(initial_population[0],num_jobs,num_machines,machine_matrix,duration_matrix)
    parent1 = tournament_selection(initial_population,2,num_jobs,num_machines,machine_matrix,duration_matrix)
    parent2 = tournament_selection(initial_population,2,num_jobs,num_machines,machine_matrix,duration_matrix) 
    child1, child2 = crossover(parent1,parent2,num_jobs,num_machines,0.6)

    print()

if __name__ == "__main__":
    main()
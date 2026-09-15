import random
import time

# random.seed(42)

class GeneticAlgorithm:
    '''
    Genetic algorithm solver for the Job Shop Scheduling Problem (JSSP).
    '''
    def __init__(
        self,
        instance,
        population_size = 100,
        generations     = 1250,
        mutation_rate   = 0.05,
        crossover_rate  = 0.8,
        tournament_size = 2,
    ):
        '''
        Initializes the genetic algorithm with problem instance and parameters.
        '''
        self.num_jobs         = instance[0]
        self.num_machines     = instance[1]
        self.jobs             = instance[2]
        self.population_size  = population_size
        self.generations      = generations
        self.mutation_rate    = mutation_rate
        self.crossover_rate   = crossover_rate
        self.tournament_size  = tournament_size


    def create_chromosome(self):
        '''
        Creates a random chromosome for the job shop scheduling problem.
        Each job is represented by its ID, and the chromosome is a list of job IDs.
        Each job ID appears in the chromosome as many times as it has operations 
        (equal to the number of machines).
        '''
        chromosome = []

        for job_id in range(self.num_jobs):
            chromosome.extend([job_id] * self.num_machines)

        random.shuffle(chromosome)

        return chromosome 


    def decode(self, chromosome, return_schedule=False):
        '''
        Decodes a chromosome using a semi-active schedule builder.
        Checks when the previous operation is done and when the required machine is ready.
        The maximum of these two is then the start time where both of these are fulfilled.
        The schedule is represented as a list of dictionaries, where each dictionary contains:
        - job: the job ID
        - operation: the operation ID (0 to NUM_MACHINES-1)
        - machine: the machine ID for this operation
        - start: the start time of the operation
        - finish: the finish time of the operation
        Returns the makespan, and optionally the schedule.
        '''
        job_counters = [0] * self.num_jobs

        job_ready_times = [0] * self.num_jobs
        machine_ready_times = [0] * self.num_machines

        schedule = []

        for job_id in chromosome:
            operation_id = job_counters[job_id]

            machine, processing_time = self.jobs[job_id][operation_id]

            start_time = max(
                job_ready_times[job_id],
                machine_ready_times[machine]
            )
            finish_time = start_time + processing_time

            if return_schedule:
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

        if return_schedule:
            return schedule, makespan

        return makespan


    def fitness(self, chromosome):
        '''
        Wrapper for decode(), avoiding cost of building schedule dictionaries.
        Returns the computed makespan.
        '''
        return self.decode(chromosome)

    def build_schedule(self, chromosome):
        '''
        Another wrapper for decode().
        Returns the schedule along with it's computed makespan.
        '''
        return self.decode(chromosome, return_schedule=True)


    def create_population(self):
        '''
        Create an initial population of chromosomes.
        '''
        population = [
            self.create_chromosome()
            for _ in range(self.population_size)
        ]
        
        population_fitness_list = []
        for chromosome in population:
            fitness_value = self.fitness(chromosome)
            population_fitness_list.append([chromosome, fitness_value])

        return population_fitness_list


    def tournament_selection(self, population):
        '''
        Selects the best individual from a random tournament sample.
        '''
        candidate_list = random.sample(
            population,
            self.tournament_size
        )

        return min(candidate_list, key=lambda item: item[1])


    def crossover_pox(self, parent1, parent2):
        '''
        Applies Precedence Preserving Order-based Crossover (POX).
        '''
        list_len = len(parent1)
        child1 = [None] * list_len
        child2 = [None] * list_len

        selected = set(random.sample(range(self.num_jobs), self.num_jobs // 2))

        remaining_for_child1 = []
        for job in parent2:
            if job not in selected:
                remaining_for_child1.append(job)

        remaining_for_child2 = []
        for job in parent1:
                if job not in selected:
                    remaining_for_child2.append(job)

        for index in range(list_len):
            if parent1[index] in selected:
                child1[index] = parent1[index]
            if parent2[index] in selected:
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

    def mutate(self, chromosome):
        '''
        Mutates a chromosome using insert (75%) or swap (25%).
        '''
        mutant = chromosome.copy()
        i, j = random.sample(range(len(mutant)), 2)
        
        if random.random() < 0.75:
            gene = mutant.pop(i)
            mutant.insert(j, gene)
        else:
            mutant[i], mutant[j] = mutant[j], mutant[i]
            
        return mutant

    def run(self, verbose=True):
        '''
        Executes the GA and returns (best_solution, history).
        '''
        population_fitness_list = self.create_population()

        best_entry = min(population_fitness_list, key=lambda item: item[1])

        best_entry = [best_entry[0].copy(), best_entry[1]]
        history = []

        for generation in range(1, self.generations + 1):
            generation_best = min(population_fitness_list, key=lambda item: item[1])

            if generation_best[1] < best_entry[1]:
                best_entry = [generation_best[0].copy(), generation_best[1]]

            history.append(best_entry[1])

            if verbose:
                print(
                    f"Generation {generation}: Best Fitness = {best_entry[1]}"
                )

            new_population = []
            # Elitism
            # new_population.append([best_entry[0].copy(), best_entry[1]])

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population_fitness_list)
                parent2 = self.tournament_selection(population_fitness_list)

                parent_chromosome1 = parent1[0]
                parent_chromosome2 = parent2[0]

                # Crossover

                if random.random() < self.crossover_rate:
                    child_chromosome1, child_chromosome2 = self.crossover_pox(
                        parent_chromosome1,
                        parent_chromosome2
                    )
                else:
                    child_chromosome1 = parent_chromosome1.copy()
                    child_chromosome2 = parent_chromosome2.copy()

                # Mutation

                if random.random() < self.mutation_rate:
                    child_chromosome1 = self.mutate(child_chromosome1)

                if random.random() < self.mutation_rate:
                    child_chromosome2 = self.mutate(child_chromosome2)

                child1 = [child_chromosome1, self.fitness(child_chromosome1)]
                child2 = [child_chromosome2, self.fitness(child_chromosome2)]

                # Add children to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population_fitness_list = new_population

        final_best = min(
            population_fitness_list,
            key=lambda item: item[1]
        )
        if final_best[1] < best_entry[1]:
            best_entry = [final_best[0].copy(), final_best[1]]
            if history:
                history[-1] = best_entry[1]

        return best_entry, history


def load_data(filepath):
    '''
    Loads and parses a JSSP benchmark file into (num_jobs, num_machines, jobs).
    '''
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
    
    return [num_jobs, num_machines, jobs]


if __name__ == "__main__":
    start = time.perf_counter()

    instance = load_data("./data/la35.txt")
    jssp_solver = GeneticAlgorithm(instance)

    best_solution, history = jssp_solver.run()

    print("Best solution found:", best_solution[0])
    schedule, makespan = jssp_solver.build_schedule(best_solution[0])
    
    print("\nSchedule:")
    for operation in schedule:
        print(operation)
    print("Makespan:", makespan)

    finish = time.perf_counter()
    total_time_spend = finish-start

    print(f'Time spend: {total_time_spend:.6f}')

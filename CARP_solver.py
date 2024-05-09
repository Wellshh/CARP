import sys
import numpy as np
import time
import random
import copy
# args = sys.argv
# filepath = args[1]
# time_limit = args[3]
# random_seed = args[5]
filepath = "D:\downloads\CARP\CARP_samples\egl-s1-A.dat"
class Solution:
    def __init__(self,solutions,cost_list):
        self.solutions = solutions
        self.cost_list = cost_list
        self.initialize_load()
    def initialize_load(self):
        self.load_list = []
        for route in self.solutions:
            load = 0
            for task in route:
                load += task_list[task]
            self.load_list.append(load)
    def __eq__(self,other):
        #combine the solutions for each Solution
        my = {tuple(sublist) for sublist in self.solutions}
        others = {tuple(sublist) for sublist in other.solutions}
        
        return my == others
    def violation_sum(self):
        sum = 0
        for i in range(len(self.load_list)):
            sum += self.load_list[i] - capacity
        return sum
    def cost_sum(self):
        sum = 0
        for i in range(len(self.cost_list)):
            sum += self.cost_list[i]
        return sum


def remove_duplicate_tasks(solution):
    all_tasks = set()
    new_solution = []
    for route in solution:
        unique_route = []
        for task in route:
            if task not in all_tasks:
                unique_route.append(task)
                all_tasks.add(task)
        new_solution.append(unique_route)
    return new_solution
# need to modify this method to insert into other position
def SBX(father, mother): #sequence based crossover algorithm
    offspring_solutions = copy.deepcopy(father.solutions)
    R1_index = random.randint(0,len(father.solutions)-1)
    R2_index = random.randint(0,len(mother.solutions)-1)
    R1 = father.solutions[R1_index]
    offspring_cost_list = copy.deepcopy(father.cost_list)
    R2 = mother.solutions[R2_index]
    #Randomly split R1 and R2 into two subroutes
    split_point_R1 = random.randint(1,len(R1)-1)
    split_point_R2 = random.randint(1,len(R2)-1)
    R11 = R1[:split_point_R1]
    R12 = R1[split_point_R1:]
    R21 = R2[:split_point_R2]
    R22 = R2[split_point_R2:]
    #Create new routes by combining R11 and R22, and R21 and R12
    new_route = R11 + R22
    new_route = list(dict.fromkeys(new_route))
    offspring_solutions[R1_index] = new_route
    #remove duplicated tasks in new solutions
    offspring_solutions = remove_duplicate_tasks(offspring_solutions)
    #Reinsert missing tasks into the new route
    R1_new = offspring_solutions[R1_index]
    additional_cost = 0
    for task in R1:
        if task not in R1_new:
            min_cost_increase = np.inf
            best_index = None
            for i in range(len(R1_new) + 1):
                test_route = R1_new[:i] + [task] + R1_new[i:]
                # Calculate additional cost and violation of capacity constraints
                if i < len(test_route) - 1:
                    additional_cost = adj_matrix[test_route[i-1][1]][task[0]]+adj_matrix_origin[task[0]][task[1]]+adj_matrix[task[1]][test_route[i+1][0]]-adj_matrix[test_route[i-1][1]][test_route[i+1][0]]
                else:
                    additional_cost = adj_matrix[test_route[i-1][1]][task[0]]+adj_matrix_origin[task[0]][task[1]]
                if additional_cost < min_cost_increase:
                    min_cost_increase = additional_cost
                    best_index = i 
            if best_index is not None:
                R1_new.insert(best_index, task)
    offspring_solutions[R1_index] = R1_new
    offspring_cost_list[R1_index] += additional_cost
    return Solution(offspring_solutions,offspring_cost_list)












            


        
    

def path_scanning(rule_num,task_list):
    #put all the tasks into a list
    free = list(task_list.keys())
    solutions = []
    cost_list = []
    while len(free) != 0:
        cost = 0
        load = 0
        end_node = depot
        route = []
        while len(free) > 0:
            #find the nearest task
            temp = find_nearest_task(adj_matrix,end_node,free,load,capacity,task_list)
            if len(temp) == 1:
                route.append(temp[0])
                free.remove((temp[0][0],temp[0][1]))
                free.remove((temp[0][1],temp[0][0]))
                cost += adj_matrix[end_node][temp[0][0]] + adj_matrix_origin[temp[0][0]][temp[0][1]]
                load += task_list[(temp[0][0],temp[0][1])]
                end_node = temp[0][1]
            elif len(temp) == 0: #no tasks satisfy the condition, break
                break
            #there are multiple tasks satisfy the condition
            else:
                #apply to different rule number
                if rule_num == 1: #maximize the distance from the task to the depot
                    a = ()
                    max_distance = -np.inf
                    for i in temp:
                        if adj_matrix[i[1]][depot] > max_distance:
                            max_distance = adj_matrix[i[1]][depot]
                            a = i
                    route.append(a)
                    free.remove((a[0],a[1]))
                    free.remove((a[1],a[0]))
                    cost += adj_matrix[end_node][a[0]] + adj_matrix_origin[a[0]][a[1]]
                    load += task_list[(a[0],a[1])]
                    end_node = a[1]
                elif rule_num == 2:#minimize the distance from the task to the depot
                    a = ()
                    min_distance = np.inf
                    for i in temp:
                        if adj_matrix[i[1]][depot] < min_distance:
                            min_distance = adj_matrix[i[1]][depot]
                            a = i
                    route.append(a)
                    free.remove((a[0],a[1]))
                    free.remove((a[1],a[0]))
                    cost += adj_matrix[end_node][a[0]] + adj_matrix_origin[a[0]][a[1]]
                    load += task_list[(a[0],a[1])]
                    end_node = a[1]
                elif rule_num == 3:
                    # maximize the term dem(t)/sc(t), where dem(t) and sc(t) are demand 
                    # and serving cost of task t, respectively;
                    a = ()
                    max_term = -np.inf
                    for i in temp:
                        divide = task_list[(i[0],i[1])] / adj_matrix_origin[i[0]][i[1]]
                        if divide > max_term:
                            max_term = divide
                            a = i
                    route.append(a)
                    free.remove((a[0],a[1]))
                    free.remove((a[1],a[0]))
                    cost += adj_matrix[end_node][a[0]] + adj_matrix_origin[a[0]][a[1]]
                    load += task_list[(a[0],a[1])]
                    end_node = a[1]
                elif rule_num == 4:
                    #minimize the term dem(t)/sc(t);
                    a = ()
                    min_term = np.inf
                    for i in temp:
                        divide = task_list[(i[0],i[1])] / adj_matrix_origin[i[0]][i[1]]
                        if divide < min_term:
                            min_term = divide
                            a = i
                    route.append(a)
                    free.remove((a[0],a[1]))
                    free.remove((a[1],a[0]))
                    cost += adj_matrix[end_node][a[0]] + adj_matrix_origin[a[0]][a[1]]
                    load += task_list[(a[0],a[1])]
                    end_node = a[1]
                else:
                    #use rule 1) if the vehicle is less than half- full, otherwise use rule 2)
                    if load < capacity/2:
                        a = ()
                        max_distance = -np.inf
                        for i in temp:
                            if adj_matrix[i[1]][depot] > max_distance:
                                max_distance = adj_matrix[i[1]][depot]
                                a = i
                    else:
                        a = ()
                        min_distance = np.inf
                        for i in temp:
                            if adj_matrix[i[1]][depot] < min_distance:
                                min_distance = adj_matrix[i[1]][depot]
                                a = i
                    route.append(a)
                    free.remove((a[0],a[1]))
                    free.remove((a[1],a[0]))
                    cost += adj_matrix[end_node][a[0]] + adj_matrix_origin[a[0]][a[1]]
                    load += task_list[(a[0],a[1])]
                    end_node = a[1]
        cost += adj_matrix[end_node][depot]
        solutions.append(route)
        cost_list.append(cost)
    return solutions,cost_list

def find_nearest_task(adj_matrix,end_node,free,load,capacity,task_list):
    min_cost = np.inf
    result = []
    for node_from,node_to in free:
        if load + task_list[(node_from,node_to)] <= capacity:
            min_cost = min(min_cost,adj_matrix[end_node][node_from])
    for i in free:
        if adj_matrix[end_node][i[0]] == min_cost and load + task_list[(i[0],i[1])] <= capacity:
            result.append(i)
    return result

def random_choose(task_list):
    #Randomly generate initial solution
    solutions = []
    cost_list = []
    free = list(task_list.keys())
    while len(free) != 0:
        cost = 0
        load = 0
        end_node = depot
        route = []
        lock = copy.deepcopy(free)
        while len(free) > 0 and len(lock) > 0:
            rand = random.randint(0,len(lock)-1)
            task = lock[rand]
            if load + task_list[task] <= capacity:
                route.append(task)
                free.remove(task)
                lock.remove(task)
                free.remove((task[1],task[0]))
                lock.remove((task[1],task[0]))
                cost += adj_matrix[end_node][task[0]] + adj_matrix_origin[task[0]][task[1]]
                load += task_list[task]
                end_node = task[1]
            else: 
                lock.remove(task)
        cost += adj_matrix[end_node][depot]
        solutions.append(route)
        cost_list.append(cost)
    return solutions,cost_list


def calculate_cost(route):
    end_node = depot
    cost = 0
    for i in route:
        cost += adj_matrix[end_node][i[0]] + adj_matrix_origin[i[0]][i[1]]
        end_node = i[1]
    cost += adj_matrix[end_node][depot]
    return cost

def double_insertion(adj_matrix, capacity, task_list, object):
    solution = object.solutions
    route_num = len(solution)

    # Choose the first route and tasks to move
    rand1 = random.randint(0, route_num - 1)
    task_num1 = len(solution[rand1])
    rand2, rand3 = random.sample(range(task_num1), 2)

    # Choose another route to intersect
    rand4 = random.choice([r for r in range(route_num) if r != rand1])
    task_num2 = len(solution[rand4])
    rand5 = random.randint(0, task_num2)
    cnt = 0
    # Ensure it's not the same location and the capacity constraint holds
    while rand4 == rand1 and (rand5 == rand2 or rand5 == rand3) or \
            task_list[solution[rand1][rand2]] + task_list[solution[rand1][rand3]] + object.load_list[rand4] > capacity:
        if cnt >= 50:
            return False
        rand4 = random.choice([r for r in range(route_num) if r != rand1])
        task_num2 = len(solution[rand4])
        rand5 = random.randint(0, task_num2)
        cnt += 1

    # Insert the tasks to the destination
    task1 = solution[rand1][rand2]
    task2 = solution[rand1][rand3]
    task_to_insert = solution[rand4][rand5]
    object.solutions[rand4].insert(rand5, task2)
    object.solutions[rand4].insert(rand5, task1)

    # Remove the tasks from the original route
    object.solutions[rand1].remove(task1)
    object.solutions[rand1].remove(task2)

    # Update the cost of the routes
    object.cost_list[rand1] = calculate_cost(object.solutions[rand1])
    object.cost_list[rand4] = calculate_cost(object.solutions[rand4])

    # Update load
    object.load_list[rand1] -= task_list[task1] + task_list[task2]
    object.load_list[rand4] += task_list[task1] + task_list[task2]
    return True
def swap_operator(adj_matrix, capacity, task_list, object):
    solution = object.solutions
    route_num = len(solution)

    # Choose two different routes
    rand1, rand2 = random.sample(range(route_num), 2)

    # Choose tasks from each route
    task_num1 = len(solution[rand1])
    task_num2 = len(solution[rand2])
    rand3 = random.randint(0, task_num1 - 1)
    rand4 = random.randint(0, task_num2 - 1)
    cnt = 0

    # Ensure capacity constraint holds
    while task_list[solution[rand1][rand3]] - task_list[solution[rand2][rand4]] + object.load_list[rand2] > capacity or \
            task_list[solution[rand2][rand4]] - task_list[solution[rand1][rand3]] + object.load_list[rand1] > capacity:
        a = task_list[solution[rand1][rand3]] - task_list[solution[rand2][rand4]] + object.load_list[rand2]
        b = task_list[solution[rand2][rand4]] - task_list[solution[rand1][rand3]] + object.load_list[rand1]
        if cnt >= 50:
            return False
        rand1, rand2 = random.sample(range(route_num), 2)
        task_num1 = len(solution[rand1])
        task_num2 = len(solution[rand2])
        rand3 = random.randint(0, task_num1 - 1)
        rand4 = random.randint(0, task_num2 - 1)
        cnt += 1

    # Swap the tasks between the two routes
    task1 = solution[rand1][rand3]
    task2 = solution[rand2][rand4]
    object.solutions[rand1][rand3] = task2
    object.solutions[rand2][rand4] = task1

    # Update the cost of the routes
    object.cost_list[rand1] = calculate_cost(object.solutions[rand1])
    object.cost_list[rand2] = calculate_cost(object.solutions[rand2])

    # Update load
    object.load_list[rand1] += task_list[task2] - task_list[task1]
    object.load_list[rand2] += task_list[task1] - task_list[task2]
    return True

        


def single_insertion(adj_matrix,capacity,task_list,object):
    solution = object.solutions
    route_num = len(solution)
    rand1 = random.randint(0,route_num-1) #choose the first route
    task_num = len(solution[rand1])
    rand2 = random.randint(0,task_num-1) #choose the task to move
    rand3 = random.randint(0,route_num-1) #choose another route to intersect
    task_num_2 = len(solution[rand3])
    rand4 = random.randint(0,task_num_2-1) #choose the destination to intersect until it's not the same location
    cnt = 0
    while (rand3 == rand1 and rand4 == rand2) or task_list[solution[rand1][rand2]] + object.load_list[rand3] > capacity:
        if cnt >= 50:
            return False
        rand3 = random.randint(0,route_num-1)
        task_num_2 = len(solution[rand3])
        rand4 = random.randint(0,task_num_2-1)
        cnt += 1
        
    #insert the task to the destination
    task = solution[rand1][rand2]
    object.solutions[rand3].insert(rand4,task)
    #remove the task from the original route
    object.solutions[rand1].remove(task)
    #update the cost of the route
    object.cost_list[rand1] = calculate_cost(object.solutions[rand1])
    object.cost_list[rand3] = calculate_cost(object.solutions[rand3])
    #update load
    object.load_list[rand1] -= task_list[task]
    object.load_list[rand3] += task_list[task]
    return True

    




start_time = time.time()
# filepath = "sample.dat"
with open(filepath,'r') as file:
    data = file.readlines()
map = []
for line in data:
    line = line.strip()
    if line.startswith("VERTICES"):
        vertices = int(line.split(":")[1].strip())
    elif line.startswith("DEPOT"):
        depot = int(line.split(":")[1].strip())
    elif line.startswith("REQUIRED EDGES"):
        tasks = int(line.split(":")[1].strip())
    elif line.startswith("NON-REQUIRED"):
        non_tasks = int(line.split(":")[1].strip())
    elif line.startswith("VEHICLES"):
        vehicles = int(line.split(":")[1].strip())
    elif line.startswith("CAPACITY"):
        capacity = int(line.split(":")[1].strip())
    elif line.startswith("TOTAL"):
        total_cost = int(line.split(":")[1].strip())
    elif line.startswith("END"):
        break
    elif line.startswith("NODES") or line.startswith("NAME"):
        continue
    else:
        parts = line.split()
        map.append((parts[0],parts[1],parts[2],parts[3]))
adj_list = {} #adjacent list
task_list = {}
adj_matrix = np.full((vertices+1,vertices+1),np.inf)
np.fill_diagonal(adj_matrix,0)
for node_from,node_to,cost,demand in map:
    if node_from not in adj_list:
        adj_list[node_from] = []
    adj_list[node_from].append((node_to,cost,demand))
    if node_to not in adj_list:
        adj_list[node_to] = []
    adj_list[node_to].append((node_from,cost,demand))
    adj_matrix[int(node_from)][int(node_to)] = cost
    adj_matrix[int(node_to)][int(node_from)] = cost
    if int(demand) != 0:
        task_list[(int(node_from),int(node_to))] = int(demand)
        task_list[(int(node_to),int(node_from))] = int(demand)
# for node, neighbors in adj_list.items():
#     print(f"Node {node}:")
#     for neighbor in neighbors:
#         neighbor_node, cost, demand = neighbor
#         print(f"  -> Neighbour: {neighbor_node}, Cost: {cost}, Demand: {demand}")
# print(adj_matrix)
adj_matrix_origin = np.copy(adj_matrix)
#find the minimum distance between the neighbors
for k in range(1,vertices+1):
    for i in range(1,vertices+1):
        for j in range(1,vertices+1):
            if adj_matrix[i][j] > adj_matrix[i][k] + adj_matrix[k][j]:
                adj_matrix[i][j] = adj_matrix[i][k] + adj_matrix[k][j]
#path scanning to construct the primitive solution
solutions,cost_list = path_scanning(2,task_list)
sum = 0
for i in cost_list:
    sum += i
#Start MAENS Algorithm
#Initialization, use small move operators on the previous solutions, we use single insertion
population = []
psize = 30
object = Solution(solutions,cost_list)
population.append(object)
print(object.solutions)
op = swap_operator(adj_matrix,capacity,task_list,object)
print(object.solutions)
while len(population) < psize:
    if op:
        cnt = 0
        oj = Solution(solutions, cost_list)
        swap_operator(adj_matrix, capacity, task_list, oj)
        while oj in population and cnt < 50:
            swap_operator(adj_matrix, capacity, task_list, oj)
            cnt += 1
        population.append(oj) 

        if len(population) >= psize:
            break

        ntrial = 0
        while ntrial < 50:
            new_solutions, new_cost = random_choose(task_list)
            object = Solution(new_solutions, new_cost)
            if object not in population:
                population.append(object)
                break
            else:
                ntrial += 1
        if ntrial >= 50:
            break
    else:
        ntrial = 0
        while ntrial < 50:
            new_solutions, new_cost = random_choose(task_list)
            object = Solution(new_solutions, new_cost)
            if object not in population:
                population.append(object)
                break
            else:
                ntrial += 1
        if ntrial >= 50:
            break
psize = len(population)
# print(population)
#Main loop of MAENS
max_iterations = 500
opsize = 6 * psize
p_mutation = 0.2
MS_num = 2
generation = 0
small_operator_search = 10
while generation <= max_iterations:
    generation += 1
    #set intermidate population
    intermediate_population = copy.deepcopy(population)
    #Generate offspring by using SBX(sequence based crossover)
    for i in range(opsize):
        #choose parents from the current population
        father = random.choice(population)
        mother = random.choice(population)
        offspring = SBX(father,mother)
        # print(offspring.solutions)
        # Local Search
        r = random.random()
        if r < p_mutation:
            #Apply local search to offspring to generate better solution
            a1 = copy.deepcopy(offspring)
            a2 = copy.deepcopy(offspring)
            a3 = copy.deepcopy(offspring)
            cnt = 0
            while cnt < small_operator_search:
                temp_a1 = copy.deepcopy(a1)
                temp_a2 = copy.deepcopy(a2)
                temp_a3 = copy.deepcopy(a3)
                single_insertion(adj_matrix,capacity,task_list,a1)
                if a1.cost_sum() >= temp_a1.cost_sum():
                    a1 = temp_a1
                swap_operator(adj_matrix,capacity,task_list,a2)
                if a2.cost_sum() >= temp_a2.cost_sum():
                    a2 = temp_a2
                # double_insertion(adj_matrix,capacity,task_list,a3)
                # if a3.cost_sum() >= temp_a3.cost_sum():
                #     a3 = temp_a3
                cnt += 1
            best_solution = None
            best_cost_sum = np.inf
            for solution in [a1, a2]:
                current_cost_sum = solution.cost_sum()
                if current_cost_sum < best_cost_sum:
                    best_solution = solution
                    best_cost_sum = current_cost_sum
            offspring = best_solution
        population.append(offspring)
    #Sort the solution in population using stochastic ranking
    
        
    print(generation)
            #Using MS operator
            


            
            
            




        
        






    
        

 



        

            












                








    

run_time = time.time() - start_time
print(run_time)








import sys
import numpy as np
import time
# args = sys.argv
# filepath = args[1]
# time_limit = args[3]
# random_seed = args[5]
# filepath = "D:\downloads\CARP\CARP_samples\egl-e1-A.dat"

def path_scanning(rule_num):
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

start_time = time.time()
filepath = "sample.dat"
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
for node, neighbors in adj_list.items():
    print(f"Node {node}:")
    for neighbor in neighbors:
        neighbor_node, cost, demand = neighbor
        print(f"  -> Neighbour: {neighbor_node}, Cost: {cost}, Demand: {demand}")
print(adj_matrix)
adj_matrix_origin = np.copy(adj_matrix)
#find the minimum distance between the neighbors
for k in range(1,vertices+1):
    for i in range(1,vertices+1):
        for j in range(1,vertices+1):
            if adj_matrix[i][j] > adj_matrix[i][k] + adj_matrix[k][j]:
                adj_matrix[i][j] = adj_matrix[i][k] + adj_matrix[k][j]
print(adj_matrix)
print(task_list)
#path scanning to construct the primitive solution
solutions,cost_list = path_scanning(2)
print(solutions)
sum = 0
for i in cost_list:
    sum += i
print(sum)
        

            












                








    

run_time = time.time() - start_time
print(run_time)








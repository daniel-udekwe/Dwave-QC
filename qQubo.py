import numpy as np
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
from scipy.optimize import minimize

# Data provided
demand = np.array([
    [0, 1450, 780, 1256, 755, 959, 1353, 984, 1601, 1891, 513, 623, 560],
    [1450, 0, 547, 1501, 744, 1531, 1375, 743, 681, 670, 1756, 1261, 611],
    [780, 547, 0, 1635, 1256, 563, 1470, 1805, 1256, 1286, 1039, 1681, 770],
    [1256, 1501, 1635, 0, 1276, 1068, 1048, 804, 1631, 1702, 925, 1515, 684],
    [755, 744, 1256, 1276, 0, 1265, 1273, 1722, 593, 1270, 811, 1126, 1274],
    [959, 1531, 563, 1068, 1265, 0, 1287, 1572, 676, 1669, 1406, 1297, 1944],
    [1353, 1375, 1470, 1048, 1273, 1287, 0, 767, 1986, 585, 1045, 1396, 1300],
    [984, 743, 1805, 804, 1722, 1572, 767, 0, 1255, 782, 1569, 1818, 1985],
    [1601, 681, 1256, 1631, 593, 676, 1986, 1255, 0, 1034, 797, 1854, 1570],
    [1891, 670, 1286, 1702, 1270, 1669, 585, 782, 1034, 0, 1677, 1542, 1761],
    [513, 1756, 1039, 925, 811, 1406, 1045, 1569, 797, 1677, 0, 937, 1606],
    [623, 1261, 1681, 1515, 1126, 1297, 1396, 1818, 1854, 1542, 937, 0, 601],
    [560, 611, 770, 684, 1274, 1944, 1300, 1985, 1570, 1761, 1606, 601, 0],
])

free_flow_travel_time = np.array([
    [0, 3.6, np.inf, np.inf, 3, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [3.6, 0, 2.4, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, 2.4, 0, 3.6, np.inf, 2.4, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, 3.6, 0, np.inf, np.inf, 2.4, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [3, np.inf, np.inf, np.inf, 0, 2.4, np.inf, 1.2, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, 2.4, np.inf, 2.4, 0, 2.4, np.inf, np.inf, 1.2, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, 2.4, np.inf, 2.4, 0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, 1.2, np.inf, np.inf, 0, 2.4, np.inf, 3, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, 0, 3, np.inf, 2.4, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, 1.2, np.inf, np.inf, 3, 0, np.inf, np.inf, 1.2],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 3, np.inf, np.inf, 0, 1.8, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, 1.8, 0, 1.2],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.2, np.inf, 1.2, 0]
])

population_distribution = np.sum(demand, axis=1)

capacity_of_link_after_hurricane = np.zeros((13, 13))
capacity_of_link_after_hurricane[0, 1] = 6.02
capacity_of_link_after_hurricane[1, 0] = 6.02
capacity_of_link_after_hurricane[2, 3] = 12.02
capacity_of_link_after_hurricane[3, 2] = 12.02
capacity_of_link_after_hurricane[2, 5] = 46.81
capacity_of_link_after_hurricane[5, 2] = 46.81
capacity_of_link_after_hurricane[4, 5] = 46.81
capacity_of_link_after_hurricane[5, 4] = 46.81
capacity_of_link_after_hurricane[5, 6] = 25.82
capacity_of_link_after_hurricane[6, 5] = 25.82
capacity_of_link_after_hurricane[4, 7] = 28.25
capacity_of_link_after_hurricane[7, 4] = 28.25
capacity_of_link_after_hurricane[7, 8] = 13.86
capacity_of_link_after_hurricane[8, 7] = 13.86
capacity_of_link_after_hurricane[10, 11] = 15.68
capacity_of_link_after_hurricane[11, 10] = 15.68
capacity_of_link_after_hurricane[11, 12] = 46.81
capacity_of_link_after_hurricane[12, 11] = 46.81

restoration_mat = np.zeros((13, 13))
restoration_mat[0, 1] = 7.38
restoration_mat[1, 0] = 7.38
restoration_mat[2, 3] = 42.17
restoration_mat[3, 2] = 42.17
restoration_mat[2, 5] = 22.31
restoration_mat[5, 2] = 22.31
restoration_mat[4, 5] = 12.56
restoration_mat[5, 4] = 12.56
restoration_mat[5, 6] = 35.33
restoration_mat[6, 5] = 35.33
restoration_mat[4, 7] = 26.82
restoration_mat[7, 4] = 26.82
restoration_mat[7, 8] = 9.37
restoration_mat[8, 7] = 9.37
restoration_mat[10, 11] = 17.23
restoration_mat[11, 12] = 15.98
restoration_mat[12, 11] = 15.98

budget = 50

# Constants
alpha = 0.15
beta = 4
theta = 1

# Function to calculate travel time based on BPR function
def travel_time(x, c):
    return 3.6 * (1 + alpha * (x / c)**beta)

# Function to calculate the total system travel time (TSTT)
def calculate_TSTT(flow, capacity):
    TSTT = 0
    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            if capacity[i, j] > 0 and flow[i, j] > 0:
                TSTT += flow[i, j] * travel_time(flow[i, j], capacity[i, j])
    return TSTT

# Function to calculate node accessibility
def calculate_accessibility(population, travel_times):
    accessibility = np.zeros(len(population))
    for r in range(len(population)):
        for s in range(len(population)):
            if travel_times[r, s] < np.inf and travel_times[r, s] > 0:  # Handling division by zero
                accessibility[r] += population[s] / travel_times[r, s]**theta
    return accessibility

# Function to calculate GINI coefficient
def calculate_GINI(accessibility):
    n = len(accessibility)
    mean_accessibility = np.mean(accessibility)
    diff_sum = np.sum([np.abs(accessibility[i] - accessibility[j]) for i in range(n) for j in range(n)])
    GINI = diff_sum / (2 * n**2 * mean_accessibility)
    return GINI

# Compute initial TSTT before restoration
initial_flow = demand
initial_capacity = free_flow_travel_time
initial_TSTT = calculate_TSTT(initial_flow, initial_capacity)

# Compute TSTT after restoration (initial guess)
restored_capacity = capacity_of_link_after_hurricane + restoration_mat
restored_TSTT = calculate_TSTT(initial_flow, restored_capacity)

# Calculate D (recovery deficiency index)
D = 1 - initial_TSTT / restored_TSTT

# Calculate E (GINI coefficient based on node accessibility)
initial_accessibility = calculate_accessibility(population_distribution, free_flow_travel_time)
restored_accessibility = calculate_accessibility(population_distribution, restored_capacity)
E = calculate_GINI(restored_accessibility)

# Define the upper-level objective function coefficients
mu = 0.5

# Parameters for QUBO
n_bits = 4
lambda_penalty = 1000  # Penalty coefficient for constraints

# Discretize restoration matrix
discretized_restoration_mat = (restoration_mat * (2**n_bits - 1)).astype(int)

# Initialize QUBO
Q = dimod.BinaryQuadraticModel('BINARY')

# Convert variables to binary
variables = {}
for i in range(len(capacity_of_link_after_hurricane)):
    for j in range(len(capacity_of_link_after_hurricane)):
        if capacity_of_link_after_hurricane[i, j] > 0:
            for k in range(n_bits):
                var_name = f"b_{i}_{j}_{k}"
                variables[(i, j, k)] = var_name
                Q.add_variable(var_name)

# Add slack variables to the QUBO for budget constraint
slack_variables = {}
num_slack_vars = 10  # Assume we use 10 slack variables
for i in range(num_slack_vars):
    slack_var_name = f"s_{i}"
    slack_variables[i] = slack_var_name
    Q.add_variable(slack_var_name)

# Add terms to QUBO objective
for (i, j, k), var_name in variables.items():
    weight = discretized_restoration_mat[i, j] / (2**k)
    Q.set_linear(var_name, mu * weight * D + (1 - mu) * weight * E)

# Add budget constraint as an equality constraint using slack variables
budget_terms = [(variables[(i, j, k)], 1.0) for (i, j, k) in variables.keys()] + [(slack_variables[i], -2**i) for i in range(num_slack_vars)]
Q.add_linear_equality_constraint(budget_terms, lagrange_multiplier=lambda_penalty, constant=budget)

# Define the QUBO model
qubo, offset = Q.to_qubo()

# Solve the QUBO using D-Wave's sampler
sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample_qubo(qubo)

# Extract the best solution
best_solution = sampleset.first.sample

# Decode the binary solution to get the restored capacity
optimal_restored_capacity = np.zeros_like(capacity_of_link_after_hurricane)
for (i, j, k), var_name in variables.items():
    if best_solution[var_name] == 1:
        optimal_restored_capacity[i][j] += 2**k

# Ensure the total cost does not exceed the budget
total_cost = 0
for i in range(len(capacity_of_link_after_hurricane)):
    for j in range(len(capacity_of_link_after_hurricane)):
        if optimal_restored_capacity[i, j] > restoration_mat[i, j]:
            optimal_restored_capacity[i, j] = restoration_mat[i, j]
        total_cost += optimal_restored_capacity[i, j]

# Adjust capacities to ensure total cost does not exceed budget
if total_cost > budget:
    remaining_budget = budget
    for i in range(len(capacity_of_link_after_hurricane)):
        for j in range(len(capacity_of_link_after_hurricane)):
            if optimal_restored_capacity[i, j] > 0:
                if remaining_budget >= optimal_restored_capacity[i, j]:
                    remaining_budget -= optimal_restored_capacity[i, j]
                else:
                    optimal_restored_capacity[i, j] = remaining_budget
                    remaining_budget = 0

# Function to minimize TSTT using User Equilibrium (UE)
def user_equilibrium(flow, capacity):
    def obj_func(f):
        return calculate_TSTT(f.reshape(flow.shape), capacity)
    
    constraints = []
    for r in range(len(demand)):
        for s in range(len(demand)):
            if r != s and demand[r][s] > 0:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda f, r=r, s=s: np.sum(f.reshape(flow.shape)[r]) - demand[r][s]
                })
    
    bounds = [(0, np.inf) for _ in range(flow.size)]
    result = minimize(obj_func, flow.flatten(), bounds=bounds, constraints=constraints)
    return result.x.reshape(flow.shape)

# Compute the optimal traffic flow using User Equilibrium (UE) with the optimal restored capacity
optimal_flow = user_equilibrium(initial_flow, optimal_restored_capacity)
optimal_restored_TSTT = calculate_TSTT(optimal_flow, optimal_restored_capacity)
optimal_D = 1 - initial_TSTT / optimal_restored_TSTT
optimal_restored_accessibility = calculate_accessibility(population_distribution, optimal_restored_capacity)
optimal_E = calculate_GINI(optimal_restored_accessibility)
optimal_R = mu * optimal_D + (1 - mu) * optimal_E

# Output the solution
print("Optimized recovered capacity for each link:")
links = [(0, 1), (1, 0), (2, 3), (3, 2), (2, 5), (5, 2), (4, 5), (5, 4), (5, 6), (6, 5), (4, 7), (7, 4), (7, 8), (8, 7), (10, 11), (11, 10), (11, 12), (12, 11)]
for (i, j) in links:
    print(f"Link ({i+1}, {j+1}): {optimal_restored_capacity[i][j]}")

print("\nOptimal values:")
print(f"R: {optimal_R}")
print(f"D: {optimal_D}")
print(f"E: {optimal_E}")
print(f"Total cost: {np.sum(optimal_restored_capacity)}")

# Print solve time details
print("\nSolve time details:")
timing_info = sampleset.info.get('timing', {})
#print("Timing information:", timing_info)
print(f"QPU access time: {timing_info.get('qpu_access_time', 'N/A')} microseconds")
print(f"QPU programming time: {timing_info.get('qpu_programming_time', 'N/A')} microseconds")
print(f"QPU sampling time: {timing_info.get('qpu_sampling_time', 'N/A')} microseconds")
print(f"QPU anneal time per sample: {timing_info.get('qpu_anneal_time_per_sample', 'N/A')} microseconds")
print(f"QPU readout time per sample: {timing_info.get('qpu_readout_time_per_sample', 'N/A')} microseconds")
print(f"QPU access overhead time: {timing_info.get('qpu_access_overhead_time', 'N/A')} microseconds")
print(f"QPU programming time: {timing_info.get('qpu_programming_time', 'N/A')} microseconds")
print(f"QPU delay time per sample: {timing_info.get('qpu_delay_time_per_sample', 'N/A')} microseconds")
print(f"post overhead time: {timing_info.get('post_processing_overhead_time', 'N/A')} microseconds")
print(f"total post processing time: {timing_info.get('total_post_processing_time', 'N/A')} microseconds")


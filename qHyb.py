import numpy as np
import dimod
from dwave.system import LeapHybridCQMSampler
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
    [0, 3.6, np.inf, np.inf, 3, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],    # 1
    [3.6, 0, 2.4, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],  # 2
    [np.inf, 2.4, 0, 3.6, np.inf, 2.4, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],     # 3
    [np.inf, np.inf, 3.6, 0, np.inf, np.inf, 2.4, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],  # 4
    [3, np.inf, np.inf, np.inf, 0, 2.4, np.inf, 1.2, np.inf, np.inf, np.inf, np.inf, np.inf],       # 5
    [np.inf, np.inf, 2.4, np.inf, 2.4, 0, 2.4, np.inf, np.inf, 1.2, np.inf, np.inf, np.inf],        # 6
    [np.inf, np.inf, np.inf, 2.4, np.inf, 2.4, 0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],  # 7
    [np.inf, np.inf, np.inf, np.inf, 1.2, np.inf, np.inf, 0, 2.4, np.inf, 3, np.inf, np.inf],       # 8
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, 0, 3, np.inf, 2.4, np.inf],       # 9
    [np.inf, np.inf, np.inf, np.inf, np.inf, 1.2, np.inf, np.inf, 3, 0, np.inf, np.inf, 1.2],       # 10
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 3, np.inf, np.inf, 0, 1.8, np.inf],    # 11
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, 1.8, 0, 1.2],     # 12
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.2, np.inf, 1.2, 0]   # 13
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
    for i in range(len(flow)):
        for j in range(len(flow)):
            if capacity[i][j] > 0 and flow[i][j] > 0:
                TSTT += flow[i][j] * travel_time(flow[i][j], capacity[i][j])
    return TSTT

# Function to calculate node accessibility
def calculate_accessibility(population, travel_times):
    accessibility = np.zeros(len(population))
    for r in range(len(population)):
        for s in range(len(population)):
            if travel_times[r][s] < np.inf and travel_times[r][s] > 0:  # Handling division by zero
                accessibility[r] += population[s] / travel_times[r][s]**theta
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

# Create a CQM model
cqm = dimod.ConstrainedQuadraticModel()

# Add variables for the capacity to be restored on each link
num_nodes = len(capacity_of_link_after_hurricane)
variables = {}
for i in range(num_nodes):
    for j in range(num_nodes):
        if capacity_of_link_after_hurricane[i, j] > 0:
            var_name = f"c_{i}_{j}"
            variables[(i, j)] = dimod.Real(var_name)
            # Adding bounds for each variable
            cqm.add_constraint(variables[(i, j)] <= restoration_mat[i, j])
            cqm.add_constraint(variables[(i, j)] >= 0)

# Add the objective function
objective = sum(mu * D * variables[var] + (1 - mu) * E * variables[var] for var in variables)
cqm.set_objective(objective)

# Add the budget constraint
budget_constraint = sum(variables[var] for var in variables)
cqm.add_constraint(budget_constraint <= budget, label='budget_constraint')

# Solve the CQM using D-Wave's hybrid solver
sampler = LeapHybridCQMSampler()
sampleset = sampler.sample_cqm(cqm)

# Extract the best solution
best_solution = sampleset.first.sample

# Compute optimal R, D, E using the best restoration plan
optimal_restored_capacity = np.zeros_like(capacity_of_link_after_hurricane)
total_cost = 0
for var, value in best_solution.items():
    if value > 0:
        i, j = map(int, var.split('_')[1:])
        if total_cost + value <= budget:
            optimal_restored_capacity[i][j] = value
            total_cost += value

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
print(f"Total cost: {total_cost}")

# Print solve time details
print("\nSolve time details:")
print(f"QPU access time: {sampleset.info.get('qpu_access_time', 'N/A')} microseconds")
print(f"Charge time: {sampleset.info.get('charge_time', 'N/A')} microseconds")
print(f"Run time: {sampleset.info.get('run_time', 'N/A')} microseconds")
#print(f"Total solve time: {sampleset.info.get('total_real_time', 'N/A')} microseconds")

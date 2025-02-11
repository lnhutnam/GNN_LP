import os
import re
import random
import torch
import numpy as np
from scipy.optimize import linprog


def random_lp_generator(num_variables, num_constraints, nnz=100, prob_equal=0.7):
    """Function random_lp_generator

    Parameters
    ----------
    num_variables
        (int) `n`: number of variables
    num_constraints
        (int) `m`: number of constraints
    nnz
        (int) number of nonzero elements in A
    prob_equal
        (float) the probability that a constraint is a equality constraint
    """

    # randomly sample a LP problem
    # min c^T x
    # s.t. Aub x <= bub, Aeq x = beq, lb <= x <= ub
    ## Generates random coefficients for the objective function
    c = np.random.uniform(-1, 1, num_variables) * 0.01

    ## Generates the vector of independent terms (right-hand side) of the constraints
    b = np.random.uniform(-1, 1, num_constraints)

    # Generates the coefficient matrix of the constraints
    A = np.zeros((num_constraints, num_variables))
    edge_index = np.zeros((nnz, 2))
    edge_index_1d = random.sample(range(num_constraints * num_variables), nnz)
    edge_feature = np.random.normal(0, 1, nnz)
    
    for l in range(nnz):
        i = int(edge_index_1d[l] / num_variables)
        j = edge_index_1d[l] - i * num_variables
        edge_index[l, 0] = i
        edge_index[l, 1] = j
        A[i, j] = edge_feature[l]

    # Generates the limits of the decision variables
    bounds = np.random.normal(0, 10, size=(num_variables, 2))

    for j in range(num_variables):
        if bounds[j, 0] > bounds[j, 1]:
            temp = bounds[j, 0]
            bounds[j, 0] = bounds[j, 1]
            bounds[j, 1] = temp

    # Generates the type of each constraint (0 for <= constraint, 1 for = constraint)
    # circ = np.random.binomial(1, prob_equal, size = num_constraints)
    # A_ub = A[circ == 0, :]
    # b_ub = b[circ == 0]
    # A_eq = A[circ == 1, :]
    # b_eq = b[circ == 1]
    constraint_types = np.random.choice(
        [0, 1], size=num_constraints, p=[prob_equal, 1 - prob_equal]
    )

    # Ensures that at least one feasible solution exists for each equality constraint individually
    for i in range(num_constraints):
        if constraint_types[i] == 1:  # Equality constraint
            b[i] = np.dot(A[i], np.random.rand(num_variables))

    return c, A, b, constraint_types, bounds, edge_index, edge_feature


def solve_lp(c, A, b, constraint_types, bounds):
    """_summary_

    Parameters
    ----------
    c
        (np.array): random coefficients for the objective function
    A
        (np.array): coefficient matrix of the constraints
    b
        (np.array): vector of independent terms (right-hand side) of the constraints
    constraint_types
        (np.array): type of each constraint (0 for <= constraint, 1 for = constraint)
    bounds
        (np.array): limits of the decision variables

    Returns
    -------
        Return the solution to the problem and if a feasible solution was found
    """
    result = linprog(
        c,
        A_ub=A[constraint_types == 0],
        b_ub=b[constraint_types == 0],
        A_eq=A[constraint_types == 1],
        b_eq=b[constraint_types == 1],
        bounds=bounds,
    )
    # Return the solution to the problem and if a feasible solution was found
    return result.x, result.status


# c, A, b, constraint_types, bounds, EdgeIndex = random_lp_generator(3, 2, 1, 0.8)
# x, status = solve_lp(c, A, b, constraint_types, bounds)

folder = "./data/"

def lp_generator(
    num_problems, num_variables, num_constraints, out_func, nnz=100, prob_equal=0.7
):
    batches_cs = [None] * num_problems
    batches_bs = [None] * num_problems
    batches_As = [None] * num_problems
    batches_edge_indexes = [None] * num_problems
    batches_constraint_types = [None] * num_problems
    batches_lower_bounds = [None] * num_problems
    batches_upper_bounds = [None] * num_problems
    batches_solutions = [None] * num_problems
    batches_feasibilities = [None] * num_problems

    # Check the feasibility of the problem: feasible and non-feasible problems
    if out_func == "feas":
        for p in range(num_problems):
            path = folder + "/problem" + str(p)
            if not os.path.exists(path):
                os.makedirs(path)
            
            c, A, b, constraint_types, bounds, edge_index, edge_feature = random_lp_generator(
                num_variables, num_constraints, nnz, prob_equal
            )
            solution, feasibility = solve_lp(c, A, b, constraint_types, bounds)

            lower_bounds, upper_bounds = zip(*bounds)

            batches_cs[p] = c
            batches_bs[p] = b
            batches_As[p] = A
            batches_constraint_types[p] = constraint_types
            batches_lower_bounds[p] = lower_bounds
            batches_upper_bounds[p] = upper_bounds
            batches_edge_indexes[p] = edge_index

            if type(solution) != type(None):
                batches_solutions[p] = solution
            else:
                batches_solutions[p] = np.zeros(num_variables)

            batches_feasibilities[p] = feasibility
            
            np.savetxt(os.path.join(path, "con_features.csv"), 
                       np.hstack((b.reshape(num_constraints, 1), constraint_types.reshape(num_constraints, 1))), 
                       delimiter=',', fmt='%10.5f')
            np.savetxt(os.path.join(path, "edge_features.csv"), edge_feature, fmt='%10.5f')
            np.savetxt(os.path.join(path, "edge_indices.csv"), edge_index, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(path, "var_features.csv"), 
                       np.hstack((c.reshape(num_variables, 1), bounds)), 
                       delimiter=',', fmt='%10.5f')
            np.savetxt(os.path.join(path, "labels_feas.csv"), [feasibility], fmt='%d')

    # To verify the objective value and solution of the problem, the generated data contains only feasible problems
    else:
        p = 0
        while p < num_problems:
            path = folder + "/problem" + str(p)
            if not os.path.exists(path):
                os.makedirs(path)
            
            c, A, b, constraint_types, bounds, edge_index, edge_feature = random_lp_generator(
                num_variables, num_constraints, nnz, prob_equal
            )
            solution, feasibility = solve_lp(c, A, b, constraint_types, bounds)

            if type(solution) == type(None):
                continue

            lower_bounds, upper_bounds = zip(*bounds)

            batches_cs[p] = c
            batches_bs[p] = b
            batches_As[p] = A
            batches_constraint_types[p] = constraint_types
            batches_lower_bounds[p] = lower_bounds
            batches_upper_bounds[p] = upper_bounds
            batches_edge_indexes[p] = edge_index

            if type(solution) != type(None):
                batches_solutions[p] = solution
            else:
                batches_solutions[p] = np.zeros(num_variables)

            batches_feasibilities[p] = feasibility
            
            np.savetxt(os.path.join(path, "con_features.csv"), 
                       np.hstack((b.reshape(num_constraints, 1), constraint_types.reshape(num_constraints, 1))), 
                       delimiter=',', fmt='%10.5f')
            np.savetxt(os.path.join(path, "edge_features.csv"), edge_feature, fmt='%10.5f')
            np.savetxt(os.path.join(path, "edge_indices.csv"), edge_index, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(path, "var_features.csv"), 
                       np.hstack((c.reshape(num_variables, 1), bounds)), 
                       delimiter=',', fmt='%10.5f')
            np.savetxt(os.path.join(path, "labels_feas.csv"), [feasibility], fmt='%d')

            p += 1

# lp_generator(50, 50, 10, 'feas')

def load_from_file(base_path = "./data"):
    
    batches_cs = []
    batches_bs = []
    batches_As = []
    batches_edge_indexes = []
    batches_constraint_types = []
    batches_lower_bounds = []
    batches_upper_bounds = []
    batches_solutions = []
    batches_feasibilities = []
    
    problem_dirs = sorted(
        [d for d in os.listdir(base_path) if re.match(r"problem\d+", d)]
    )
    
    for problem_dir in problem_dirs:
        problem_path = os.path.join(base_path, problem_dir)
        
        con_features = np.loadtxt(os.path.join(problem_path, "con_features.csv"), delimiter=",")
        batches_bs.append(torch.tensor(con_features[:, 0], dtype=torch.float32))
        batches_constraint_types.append(torch.tensor(con_features[:, 1], dtype=torch.float32))
        
        edge_features = np.loadtxt(os.path.join(problem_path, "edge_features.csv"))
        batches_edge_indexes.append(torch.tensor(edge_features, dtype=torch.float32))
        
        edge_indices = np.loadtxt(os.path.join(problem_path, "edge_indices.csv"), delimiter=",")
        batches_edge_indexes.append(torch.tensor(edge_indices, dtype=torch.long))
        
        var_features = np.loadtxt(os.path.join(problem_path, "var_features.csv"), delimiter=",")
        batches_cs.append(torch.tensor(var_features[:, 0], dtype=torch.float32))
        batches_lower_bounds.append(torch.tensor(var_features[:, 1], dtype=torch.float32))
        batches_upper_bounds.append(torch.tensor(var_features[:, 2], dtype=torch.float32))
        
        labels = np.loadtxt(os.path.join(problem_path, "labels_feas.csv"))
        batches_feasibilities.append(torch.tensor(labels, dtype=torch.float32))
    
    return {
        "batches_cs": batches_cs,
        "batches_bs": batches_bs,
        "batches_As": batches_As,
        "batches_edge_indexes": batches_edge_indexes,
        "batches_constraint_types": batches_constraint_types,
        "batches_lower_bounds": batches_lower_bounds,
        "batches_upper_bounds": batches_upper_bounds,
        "batches_solutions": batches_solutions,
        "batches_feasibilities": batches_feasibilities,
    }


# lp_data = load_from_file()
# print(f"Loaded data for {len(lp_data['batches_cs'])} problems.")
"""
    All utilites are borrowed from ECE 457A Fall 2023 Tutorials
"""

import random
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Callable, Optional
import matplotlib.pyplot as plt


def schwefel_cost(x: List[float]) -> float:
    d = len(x)
    f = 418.9829 * d
    for xi in x:
        f = f - (xi * np.sin(np.sqrt(np.abs(xi))))
    return f


def bound_solution_in_x_range(
        x: List[float],
        x_range: List[List[float]]
    ) -> List[float]:

    for j in range(len(x)):
        if x[j] < x_range[j][0]:
            x[j] = x_range[j][0]
        elif x[j] > x_range[j][1]:
            x[j] = x_range[j][1]
    return x


def local_search(
        cost_function: Callable,
        max_itr: int,
        convergence_threshold: float, 
        x_initial: Optional[np.array] = None,
        x_range: Optional[List[List[float]]] = None,
        hide_progress_bar: Optional[bool] = False
    ) -> Tuple[np.array, float, List[np.array], List[float]]:

    # Set the x_initial
    if x_initial is None:
        x_initial = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]

    x_current = x_initial
    cost_current = cost_function(x_current)

    x_history = [x_current]
    cost_history = [cost_current]

    # Create a tqdm progress bar
    if not hide_progress_bar:
        progress_bar = tqdm(total=max_itr, desc='Iterations')

    convergence = False
    itr = 0
    while not convergence:
        # Generate neighboring solutions
        x_neighbor = [random.gauss(x, 0.1) for x in x_current]
        x_neighbor = bound_solution_in_x_range(x=x_neighbor, x_range=x_range)
        cost_neighbor = cost_function(x_neighbor)

        # Accept the neighbor if it has lower cost
        if cost_neighbor < cost_current:
            x_current = x_neighbor
            cost_current = cost_neighbor

        # moved exit condition check to outside if statement, only difference from tutorial
        if (cost_current < convergence_threshold) or (itr >= max_itr):
            convergence = True

        x_history.append(x_current)
        cost_history.append(cost_current)

        # Update the tqdm progress bar
        if not hide_progress_bar:
            progress_bar.update(1)  # Increment the progress bar by 1 unit
        itr += 1
    
    # progress_bar.close()

    # Get the best solution
    best_cost_index = np.argmin(cost_history)
    best_x = x_history[best_cost_index]
    best_cost = cost_history[best_cost_index]

    return best_x, best_cost, x_history, cost_history


# if __name__ == "__main__":
#     local_search(
#         cost_function=schwefel_cost,
#         max_itr=500,
#         convergence_threshold=0.01, # the global minimum is roughly 0 for this cost fn
#         x_range=[[-500,500] for _ in range(1)] # domain is [-500,500] for each dimension of this cost fn
#     )

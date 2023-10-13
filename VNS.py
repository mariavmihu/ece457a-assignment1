import numpy as np
from typing import List, Callable, Tuple

from utils import local_search, schwefel_cost
from plot_utils import plot_results


def create_partitions(num_neighbourhoods: int, x_range: List[List[float]] = None):

    # Evenly partition the max_range into num_neighborhoods
    neighborhoods = []
    for i in range(num_neighbourhoods):
        neighborhood = [
            x_range[0][0] + i * ((x_range[0][1] - x_range[0][0]) / num_neighbourhoods),
            x_range[0][0] + (i + 1) * ((x_range[0][1] - x_range[0][0]) / num_neighbourhoods)
        ]

        neighborhood_with_dimensions = []
        for j in range(len(x_range)):
            if j != len(x_range) - 1 or len(x_range) == 1:
                # section off the first D-1 dimensions
                neighborhood_with_dimensions.append(neighborhood)
            else:
                # for the last dimension, partition includes all points
                neighborhood_with_dimensions.append(x_range[0])

        neighborhoods.append(neighborhood_with_dimensions)

    return neighborhoods


def VNS_search(
    cost_function: Callable,
    max_itr: int,
    max_itr_local: int,
    convergence_threshold: float,
    num_neighbourhoods: int,
    x_range: List[List[float]], 
)-> Tuple[np.array, float, List[np.array], List[float]]:
    # args NOTE: choosing not to implement x_initial, and choosing NOT to make x_range Optional

    # local search over entire domain
    global_best_x, global_best_cost, x_history, cost_history = local_search(
        cost_function=schwefel_cost,
        max_itr=max_itr_local,
        convergence_threshold=convergence_threshold,
        x_range=x_range,
        hide_progress_bar=True
    )

    # partition into neighbourhoods
    # 1D NBHs:
    # [ [ [-500.0, -300.0] ], [ [-300.0, -100.0] ], [ [-100.0, 100.0] ], [ [100.0, 300.0] ], [ [300.0, 500.0] ] ]
    # 2D NBHs:
    # [ N1:[[-500.0, -300.0], [-500, 500]], N2:[[-300.0, -100.0], [-500, 500]], N3:[[-100.0, 100.0], [-500, 500]], 
    #  N4:[[100.0, 300.0], [-500, 500]], N5:[[300.0, 500.0], [-500, 500]] ]
    neighbourhoods = create_partitions(num_neighbourhoods, x_range)

    # alternating neighbourhood logic
    # for i, N in enumerate(neighbourhoods):
    convergence = False
    i = 0 # starting with the first neighbourhood --> #NOTE: could randomize for better exploration?
    itr = 1
    while not convergence:
        local_best_x, local_best_cost, incoming_x_history, incoming_cost_history = local_search(
            cost_function=cost_function,
            max_itr=max_itr_local,
            convergence_threshold=0.01,
            x_range=neighbourhoods[i],
            hide_progress_bar=True
        )

        if local_best_cost < global_best_cost:
            global_best_cost = local_best_cost
            global_best_x = local_best_x
            i = 0
        else:
            i = (i+1) if i != len(neighbourhoods)-1 else 0

        if (global_best_cost < convergence_threshold) or (itr >= max_itr):
            convergence = True

        x_history.extend(incoming_x_history)
        cost_history.extend(incoming_cost_history)
        itr += 1

    # sort out "bests" and return
    best_cost_index = np.argmin(cost_history)
    best_x = x_history[best_cost_index]
    best_cost = cost_history[best_cost_index]

    return best_x, best_cost, x_history, cost_history


if __name__ == "__main__":

    neighbourhoods = input("Please indicate how many Neighbourhoods to search, as an integer: ")
    dimensions = input("Please indicate the dimensionality, d, of the Schwefel function being optimized: ")

    best_x, best_cost, x_history, cost_history = VNS_search(
        cost_function=schwefel_cost,
        max_itr=1000,
        max_itr_local=1000,
        convergence_threshold=0.01, # the global minimum is roughly 0 for this cost fn
        num_neighbourhoods=int(neighbourhoods),
        x_range=[[-500,500] for _ in range(int(dimensions))], # domain is [-500,500] for each dimension of this cost fn
    )

    print(f"best x: {best_x}")
    print(f"best cost: {best_cost}")

    plot_results(
        best_x=best_x,
        best_cost=best_cost,
        x_history=x_history,
        cost_history=cost_history,
        cost_function=schwefel_cost,
        x_range=[[-500,500] for _ in range(int(dimensions))]
    )


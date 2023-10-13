import numpy as np
import multiprocessing
from functools import partial
from typing import List, Callable, Optional

from utils import schwefel_cost
from plot_utils import plot_results
from VNS import VNS_search


def create_regions(num_regions: int, x_range: List[List[float]], dims: int) -> List[List[List[float]]]:
    # 1D: x_range = [[-500, 500]]
    # 2D: x_range = [[-500, 500], [-500, 500]]
    # 3D: x_range = [[-500, 500], [-500, 500], [-500, 500]]

    # 2D after split for num_regions = 4: 
    # [ [ [-125,125],[-125,125] ], [ [-250,250],[-250,250] ], [ [-375,375],[-375,375] ] , [ [-500,500],[-500,500] ] ]

    domain_splits = [[(x_range[0][0]/num_regions)*(i+1),(x_range[0][1]/num_regions)*(i+1)] for i in range(num_regions)]
    regions = [[domain] * dims for domain in domain_splits]

    return regions


def VNS_search_wrapper(args):
    cost_function, max_itr, max_itr_local, convergence_threshold, num_neighbourhoods, x_range = args
    return VNS_search(
        cost_function=cost_function,
        max_itr=max_itr,
        max_itr_local=max_itr_local,
        convergence_threshold=convergence_threshold,
        num_neighbourhoods=num_neighbourhoods,
        x_range=x_range
    )


def GNS_search(
    cost_function: Callable,
    max_itr: int,
    max_itr_local: int,
    convergence_threshold: float,
    num_regions: int,
    num_neighbourhoods: int,
    x_range: List[List[float]], 
    dims:int,
):

    best_x, best_cost, x_history, cost_history = None, None, [], []

    num_processes = num_regions
    x_range_regions = create_regions(num_regions, x_range, dims)

    VNS_inputs = [
        (
            cost_function,
            max_itr,
            max_itr_local,
            convergence_threshold,
            num_neighbourhoods,
            region,
        ) for region in x_range_regions
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        VNS_results = pool.map(partial(VNS_search_wrapper), VNS_inputs)

    for (_, _, vns_x_history, vns_cost_history) in VNS_results:
        x_history.extend(vns_x_history)
        cost_history.extend(vns_cost_history)

    # sort out "bests" and return
    best_cost_index = np.argmin(cost_history)
    best_x = x_history[best_cost_index]
    best_cost = cost_history[best_cost_index]

    return best_x, best_cost, x_history, cost_history


if __name__ == "__main__":

    regions = input("Please indicate how many regions to perform GNS on, as an integer: ")
    neighbourhoods = input("Please indicate how many Neighbourhoods to search, as an integer: ")
    dimensions = input("Please indicate the dimensionality, d, of the Schwefel function being optimized: ")

    best_x, best_cost, x_history, cost_history = GNS_search(
        cost_function=schwefel_cost,
        max_itr=100,
        max_itr_local=1000,
        convergence_threshold=0.01, # the global minimum is roughly 0 for this cost fn
        num_regions=int(regions),
        num_neighbourhoods=int(neighbourhoods),
        x_range=[[-500,500] for _ in range(int(dimensions))], # domain is [-500,500] for each dimension of this cost fn
        dims=int(dimensions)
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

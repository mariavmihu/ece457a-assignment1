# ECE-457A Assignment 1

## Requirements
Create a python virtual environment and run `pip install -r requirements.txt` in order to install dependencies needed for the tutorial code as well as my code.

## What each file does
### utils.py
Includes the local search code from the tutorial (with one minor change to how the exit conditions are verified), as well as the Schwefel cost function and any other utility functions from the tutorials that were needed.

### plot_utils.py
Exact copy of plotting utils from the tutorial.

### VNS.py
The full source code for my implementation of Variable Neighbourhood Search, including a `__main__` loop that can be used to call and visualize the algorithm. To run, simply run the following in the repository directory:

    ```
    python3 VNS.py
    ```
You will be prompted for user inputs in the command line; there are no additional parameters for running the script.

### GNS.py
The full source code for my implementation of Generalized Neighbourhood Search, including a `__main__` loop that can be used to call and visualize the algorithm. To run, simply run the following in the repository directory:

    ```
    python3 GNS.py
    ```
You will be prompted for user inputs in the command line; there are no additional parameters for running the script.
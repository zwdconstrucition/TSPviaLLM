# FunSearch-TSP

Implementation of the FunSearch algorithm for solving Traveling Salesman Problems (TSP) with comparison to Google OR-Tools.

## Overview

This project demonstrates how to use FunSearch (Function Search), a new approach to optimization that evolves program code to solve complex problems, and applies it to the Traveling Salesman Problem. It includes benchmarking against Google OR-Tools, a state-of-the-art optimization library.

## Directory Structure

- `implementation/`: Python modules implementing the core functionality
  - `benchmark_utils.py`: Utilities for benchmarking against OR-Tools
  - `tsp_loader.py`: Utilities for loading and processing TSP instances
  - `funsearch_integration.py`: Integration between FunSearch and OR-Tools

- `FunSearch_TSP.ipynb`: Main notebook implementing FunSearch for TSP
- `TSPviaLLM_v1.ipynb`: Notebook using LLMs to generate TSP heuristics

## Features

- Evolution of priority functions for TSP tour construction
- Integration with Google OR-Tools for benchmarking
- Visualization of tour quality progression over iterations
- Comparative analysis between FunSearch and OR-Tools

## Getting Started

1. Install the required dependencies:
   ```
   pip install numpy pandas matplotlib ortools
   ```

2. Run the `FunSearch_TSP.ipynb` notebook to see FunSearch in action

## Results

The implementation tracks various performance metrics:
- Tour length comparison between FunSearch and OR-Tools
- Performance gap percentage over iterations
- Visualization of both tours on the same map

Results are saved to the `comparison_results` directory.

## References

- FunSearch paper: [FunSearch: Making New Discoveries in Mathematical Sciences using Large Language Models](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)
- Google OR-Tools: [https://developers.google.com/optimization](https://developers.google.com/optimization)
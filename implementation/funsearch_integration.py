"""
Integration utilities to connect FunSearch with the OR-Tools benchmark.
"""

from typing import Dict, List, Any, Callable
import time
from .benchmark_utils import BenchmarkTracker


class FunSearchEvaluator:
    """
    Evaluate FunSearch-generated TSP heuristics against OR-Tools benchmarks.
    
    This class helps track and compare the performance of FunSearch-generated
    heuristics across multiple iterations and TSP instances.
    """
    
    def __init__(self, tsp_instances):
        """
        Initialize with TSP instances to benchmark against.
        
        Args:
            tsp_instances: Dictionary mapping instance names to loaded TSP data
        """
        self.tracker = BenchmarkTracker(tsp_instances)
        self.current_iteration = 0
        self.default_instance = next(iter(tsp_instances.keys()))
        
    def evaluate_heuristic(self, heuristic_func, instance_name=None):
        """
        Evaluate a heuristic function against the OR-Tools benchmark.
        
        Args:
            heuristic_func: Function that implements the TSP heuristic
            instance_name: Name of the TSP instance to evaluate against (default: first available instance)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if instance_name is None:
            instance_name = self.default_instance
            
        data = self.tracker.tsp_instances[instance_name]
        
        # Time the heuristic execution
        start_time = time.time()
        routes = heuristic_func(data['distance_matrix'], data['num_vehicles'], data['depot'])
        end_time = time.time()
        
        # Handle FunSearch heuristic returning a list of routes or just a single route
        if isinstance(routes, list) and all(isinstance(node, int) for node in routes):
            # Single route as a list of integers
            route = routes
        elif isinstance(routes, list) and len(routes) > 0:
            # List of routes, take the first one (for classic TSP)
            route = routes[0]
        else:
            raise ValueError("Invalid route format returned by heuristic function")
            
        # Track this iteration's results
        comparison = self.tracker.track_iteration(
            self.current_iteration,
            instance_name,
            route
        )
        
        # Include runtime information
        comparison['runtime'] = end_time - start_time
        
        return comparison
        
    def next_iteration(self):
        """
        Increment the iteration counter.
        """
        self.current_iteration += 1
        
    def create_summary(self):
        """
        Create a summary of the best results for each instance.
        
        Returns:
            Pandas DataFrame with summary results
        """
        return self.tracker.create_summary()
        
    def plot_progress(self, instance_name=None):
        """
        Plot the progress of the heuristic against OR-Tools over iterations.
        
        Args:
            instance_name: Name of the TSP instance (default: first available instance)
        """
        if instance_name is None:
            instance_name = self.default_instance
            
        self.tracker.plot_progress_chart(instance_name)
        
    def plot_best_comparison(self, instance_name=None):
        """
        Plot the comparison between the best heuristic and OR-Tools.
        
        Args:
            instance_name: Name of the TSP instance (default: first available instance)
        """
        if instance_name is None:
            instance_name = self.default_instance
            
        self.tracker.plot_final_comparison(instance_name)
        
    def generate_all_visualizations(self):
        """
        Generate all visualizations for all tracked instances.
        """
        for instance_name in self.tracker.tsp_instances.keys():
            self.plot_progress(instance_name)
            self.plot_best_comparison(instance_name)
            
        # Save summary as CSV
        self.tracker.save_summary()

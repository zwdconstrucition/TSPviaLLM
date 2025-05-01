"""
Benchmark utilities for comparing FunSearch heuristics with OR-Tools on TSP problems.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from typing import List, Dict, Tuple, Optional, Any
import math


def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points in 2D space."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def haversine(coord1, coord2):
    """Calculate the haversine distance between two points on the Earth given their latitude and longitude."""
    R = 6371.0  # Radius of the Earth in kilometers
    lat1 = coord1[0]
    lon1 = coord1[1]
    lat2 = coord2[0]
    lon2 = coord2[1]
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # Distance in kilometers
    return distance


def create_distance_matrix(coords, distance_type='EUC_2D'):
    """Create a distance matrix from the coordinates."""
    num_nodes = len(coords)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                if distance_type == 'EUC_2D':
                    distance_matrix[i][j] = euclidean_distance(coords[i], coords[j])
                elif distance_type == 'GEO':
                    distance_matrix[i][j] = haversine(coords[i], coords[j])
            else:
                distance_matrix[i][j] = 0.0
    return distance_matrix


def solve_tsp_with_ortools(data):
    """Solve the TSP with Google OR-Tools.
    
    Args:
        data: Dictionary containing 'distance_matrix', 'num_vehicles', 'depot', etc.
        
    Returns:
        route: List of node indices forming the optimal route found
        objective_value: The objective value from OR-Tools
        route_distance: The actual distance of the route
    """
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                          data['num_vehicles'],
                                          data['depot'])

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Create the distance callback and register it with the routing model
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['distance_matrix'][from_node][to_node] * 100)  # Scale for integer costs

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return None, None, None

    # Extract the route
    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append(node_index)
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))  # Add depot to end

    objective_value = solution.ObjectiveValue()
    route_distance = sum(data['distance_matrix'][route[i]][route[i+1]]
                         for i in range(len(route) - 1))
    route_distance += data['distance_matrix'][route[-1]][route[0]]

    return route, objective_value, route_distance


def calculate_route_distance(route, distance_matrix):
    """Calculate the total distance of a route.
    
    Args:
        route: List of node indices forming the route
        distance_matrix: 2D array of distances between nodes
        
    Returns:
        Total distance of the route
    """
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    # Add distance back to depot
    if len(route) > 1:
        total_distance += distance_matrix[route[-1]][route[0]]
    return total_distance


def benchmark_ortools(tsp_instances):
    """Run OR-Tools benchmark on multiple TSP instances.
    
    Args:
        tsp_instances: Dictionary mapping instance names to loaded TSP data
        
    Returns:
        Dictionary mapping instance names to (route, objective_value, route_distance)
    """
    results = {}
    
    for name, data in tsp_instances.items():
        print(f"Solving {name} with OR-Tools...")
        route, objective_value, route_distance = solve_tsp_with_ortools(data)
        results[name] = {
            'route': route,
            'objective_value': objective_value,
            'route_distance': route_distance,
        }
        
        if route is not None:
            print(f"  Objective value: {objective_value}")
            print(f"  Route distance: {route_distance}")
        else:
            print("  Could not find a solution.")
            
    return results


def compare_with_ortools(heuristic_route, distance_matrix, ortools_distance):
    """Compare a heuristic route against OR-Tools benchmark.
    
    Args:
        heuristic_route: Route generated by heuristic as list of node indices
        distance_matrix: Distance matrix for the TSP instance
        ortools_distance: Distance of the OR-Tools optimal route
        
    Returns:
        Dictionary with comparison metrics
    """
    heuristic_distance = calculate_route_distance(heuristic_route, distance_matrix)
    gap_percent = ((heuristic_distance - ortools_distance) / ortools_distance) * 100
    
    return {
        'heuristic_distance': heuristic_distance,
        'ortools_distance': ortools_distance,
        'absolute_gap': heuristic_distance - ortools_distance,
        'percent_gap': gap_percent,
        'is_better': gap_percent < 0
    }


def plot_comparison(coords, heuristic_route, ortools_route, title="Route Comparison", save_path=None):
    """Plot both heuristic and OR-Tools routes on the same map for visual comparison.
    
    Args:
        coords: List of coordinates for each node
        heuristic_route: Route generated by heuristic
        ortools_route: Route generated by OR-Tools
        title: Plot title
        save_path: Path to save the plot image (optional)
    """
    plt.figure(figsize=(12, 10))
    
    # Convert coords to numpy array for easier handling
    coords_np = np.array(coords)
    x_coords = coords_np[:, 0]
    y_coords = coords_np[:, 1]
    
    # Plot all cities
    plt.scatter(x_coords, y_coords, c='black', s=50, alpha=0.5, label='Cities')
    
    # Plot OR-Tools route in red
    for i in range(len(ortools_route)-1):
        start_idx = ortools_route[i]
        end_idx = ortools_route[i+1]
        plt.plot([coords[start_idx][0], coords[end_idx][0]],
                 [coords[start_idx][1], coords[end_idx][1]], 'r-', alpha=0.7)
    
    # Complete OR-Tools route back to depot
    plt.plot([coords[ortools_route[-1]][0], coords[ortools_route[0]][0]],
             [coords[ortools_route[-1]][1], coords[ortools_route[0]][1]], 'r-', alpha=0.7)
    
    # Plot heuristic route in blue
    for i in range(len(heuristic_route)-1):
        start_idx = heuristic_route[i]
        end_idx = heuristic_route[i+1]
        plt.plot([coords[start_idx][0], coords[end_idx][0]],
                 [coords[start_idx][1], coords[end_idx][1]], 'b-', alpha=0.7)
    
    # Complete heuristic route back to depot
    plt.plot([coords[heuristic_route[-1]][0], coords[heuristic_route[0]][0]],
             [coords[heuristic_route[-1]][1], coords[heuristic_route[0]][1]], 'b-', alpha=0.7)
    
    # Mark depot (starting point)
    plt.scatter(coords[0][0], coords[0][1], c='green', s=200, marker='*', label='Depot')
    
    # Add legend with custom lines
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='r', lw=2),
                   Line2D([0], [0], color='b', lw=2)]
    plt.legend(custom_lines, ['OR-Tools Route', 'Heuristic Route'])
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()


def plot_progress(iterations, gaps, title="Performance Gap Over Iterations", save_path=None):
    """Plot the evolution of performance gap over iterations.
    
    Args:
        iterations: List of iteration numbers
        gaps: List of gap percentages corresponding to iterations
        title: Plot title
        save_path: Path to save the plot image (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot gap percentage over iterations
    plt.plot(iterations, gaps, 'b-o')
    
    # Add horizontal line at 0%
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Performance Gap (%)')
    plt.grid(True)
    
    # Add annotations for starting and ending values
    if len(gaps) > 1:
        plt.annotate(f"{gaps[0]:.2f}%", (iterations[0], gaps[0]), 
                   textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f"{gaps[-1]:.2f}%", (iterations[-1], gaps[-1]), 
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()


def create_summary_table(results):
    """Create a table summarizing comparison results.
    
    Args:
        results: Dictionary mapping TSP instances to comparison results
        
    Returns:
        Pandas DataFrame with comparison results
    """
    # Create pandas DataFrame for display
    data = []
    
    for instance, result in results.items():
        data.append({
            'Instance': instance,
            'OR-Tools Distance': result['ortools_distance'],
            'FunSearch Distance': result['heuristic_distance'],
            'Gap (%)': result['percent_gap'],
            'Better than OR-Tools': 'Yes' if result['is_better'] else 'No'
        })
    
    df = pd.DataFrame(data)
    return df


class BenchmarkTracker:
    """Class to track and visualize the performance of FunSearch heuristics against OR-Tools benchmark."""
    
    def __init__(self, tsp_instances):
        """Initialize with TSP instances.
        
        Args:
            tsp_instances: Dictionary mapping instance names to loaded TSP data
        """
        self.tsp_instances = tsp_instances
        self.ortools_results = benchmark_ortools(tsp_instances)
        self.iteration_history = {name: [] for name in tsp_instances}
        self.output_dir = "comparison_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def track_iteration(self, iteration, instance_name, heuristic_route):
        """Track results for a single iteration.
        
        Args:
            iteration: Iteration number
            instance_name: Name of the TSP instance
            heuristic_route: Route generated by the heuristic
            
        Returns:
            Comparison metrics
        """
        data = self.tsp_instances[instance_name]
        ortools_result = self.ortools_results[instance_name]
        
        comparison = compare_with_ortools(
            heuristic_route, 
            data['distance_matrix'], 
            ortools_result['route_distance']
        )
        
        # Store this iteration's results
        self.iteration_history[instance_name].append({
            'iteration': iteration,
            'route': heuristic_route,
            'comparison': comparison
        })
        
        return comparison
    
    def plot_final_comparison(self, instance_name):
        """Plot the final comparison between the best heuristic and OR-Tools.
        
        Args:
            instance_name: Name of the TSP instance
        """
        history = self.iteration_history[instance_name]
        if not history:
            print(f"No history available for {instance_name}")
            return
            
        # Find the best iteration
        best_iteration = min(history, key=lambda x: x['comparison']['heuristic_distance'])
        
        # Plot comparison
        plot_comparison(
            self.tsp_instances[instance_name]['coords'],
            best_iteration['route'],
            self.ortools_results[instance_name]['route'],
            title=f"Best FunSearch vs OR-Tools: {instance_name}",
            save_path=f"{self.output_dir}/{instance_name}_route_comparison.png"
        )
        
    def plot_progress_chart(self, instance_name):
        """Plot the progress of the heuristic against OR-Tools over iterations.
        
        Args:
            instance_name: Name of the TSP instance
        """
        history = self.iteration_history[instance_name]
        if not history:
            print(f"No history available for {instance_name}")
            return
            
        iterations = [entry['iteration'] for entry in history]
        gaps = [entry['comparison']['percent_gap'] for entry in history]
        
        plot_progress(
            iterations, 
            gaps, 
            title=f"Performance Gap over Iterations: {instance_name}",
            save_path=f"{self.output_dir}/{instance_name}_progress.png"
        )
        
    def create_summary(self):
        """Create a summary of the best results for each instance.
        
        Returns:
            Pandas DataFrame with summary results
        """
        best_results = {}
        
        for name, history in self.iteration_history.items():
            if history:
                # Find the best iteration
                best_iteration = min(history, key=lambda x: x['comparison']['heuristic_distance'])
                best_results[name] = best_iteration['comparison']
                
        return create_summary_table(best_results)
        
    def save_summary(self, filename="summary.csv"):
        """Save the summary to a CSV file.
        
        Args:
            filename: Name of the CSV file
        """
        summary_df = self.create_summary()
        summary_df.to_csv(f"{self.output_dir}/{filename}", index=False)
        return summary_df

"""
Utilities to load and process TSP problem instances.
"""

import math
import numpy as np
import io
import os


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


def load_tsp_data(file_obj):
    """Load TSP data from a file-like object and return a data dictionary."""
    data = {}
    lines = file_obj.readlines()

    # Extract metadata and find edge weight type
    edge_weight_type = 'EUC_2D'  # Default value
    for line in lines[:6]:
        if 'EDGE_WEIGHT_TYPE' in line:
            edge_weight_type = line.split(':')[1].strip()

    # Extract the number of nodes
    num_nodes = 0
    for line in lines:
        if 'DIMENSION' in line:
            num_nodes = int(line.split(':')[1].strip())
            break

    # Find where the coordinates start
    start_index = 0
    for i, line in enumerate(lines):
        if 'NODE_COORD_SECTION' in line:
            start_index = i + 1
            break

    # Extract the coordinates of the nodes
    coords = []
    for line in lines[start_index:start_index+num_nodes]:
        parts = line.strip().split()
        if len(parts) >= 3:
            coords.append([float(parts[1]), float(parts[2])])

    # Create a distance matrix (assuming a Euclidean distance metric)
    data['distance_matrix'] = create_distance_matrix(coords, edge_weight_type)
    data['num_vehicles'] = 1
    data['depot'] = 0
    data['coords'] = coords
    data['num_nodes'] = num_nodes
    data['edge_weight_type'] = edge_weight_type

    return data


def load_tsp_from_string(tsp_string):
    """Load TSP data from a string and return a data dictionary."""
    return load_tsp_data(io.StringIO(tsp_string))


def load_tsp_from_file(filename):
    """Load TSP data from a file and return a data dictionary."""
    with open(filename, 'r') as f:
        return load_tsp_data(f)


DEFAULT_TSP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Create data directory if it doesn't exist
os.makedirs(DEFAULT_TSP_DIR, exist_ok=True)


def get_common_tsp_instances():
    """Return a dictionary of commonly used TSP instances.
    
    Loads TSP instances from the data directory.
    If files don't exist, downloads them from standard sources.
    """
    instances = {}
    
    # Define paths for common instances
    berlin52_path = os.path.join(DEFAULT_TSP_DIR, "berlin52.tsp")
    a280_path = os.path.join(DEFAULT_TSP_DIR, "a280.tsp")
    ch130_path = os.path.join(DEFAULT_TSP_DIR, "ch130.tsp")
    
    # Check if berlin52.tsp exists, if not, create it
    if not os.path.exists(berlin52_path):
        print(f"Creating berlin52.tsp in {DEFAULT_TSP_DIR}")
        berlin52_content = """NAME: berlin52
TYPE: TSP
COMMENT: 52 locations in Berlin (Groetschel)
DIMENSION: 52
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 565.0 575.0
2 25.0 185.0
3 345.0 750.0
4 945.0 685.0
5 845.0 655.0
6 880.0 660.0
7 25.0 230.0
8 525.0 1000.0
9 580.0 1175.0
10 650.0 1130.0
11 1605.0 620.0
12 1220.0 580.0
13 1465.0 200.0
14 1530.0 5.0
15 845.0 680.0
16 725.0 370.0
17 145.0 665.0
18 415.0 635.0
19 510.0 875.0
20 560.0 365.0
21 300.0 465.0
22 520.0 585.0
23 480.0 415.0
24 835.0 625.0
25 975.0 580.0
26 1215.0 245.0
27 1320.0 315.0
28 1250.0 400.0
29 660.0 180.0
30 410.0 250.0
31 420.0 555.0
32 575.0 665.0
33 1150.0 1160.0
34 700.0 580.0
35 685.0 595.0
36 685.0 610.0
37 770.0 610.0
38 795.0 645.0
39 720.0 635.0
40 760.0 650.0
41 475.0 960.0
42 95.0 260.0
43 875.0 920.0
44 700.0 500.0
45 555.0 815.0
46 830.0 485.0
47 1170.0 65.0
48 830.0 610.0
49 605.0 625.0
50 595.0 360.0
51 1340.0 725.0
52 1740.0 245.0
EOF"""
        with open(berlin52_path, 'w') as f:
            f.write(berlin52_content)
    
    # Load the instances from files
    if os.path.exists(berlin52_path):
        instances['berlin52'] = load_tsp_from_file(berlin52_path)
    
    if os.path.exists(a280_path):
        instances['a280'] = load_tsp_from_file(a280_path)
    
    if os.path.exists(ch130_path):
        instances['ch130'] = load_tsp_from_file(ch130_path)

    return instances


def list_available_instances():
    """List all available TSP instances in the data directory."""
    if os.path.exists(DEFAULT_TSP_DIR):
        tsp_files = [f for f in os.listdir(DEFAULT_TSP_DIR) if f.endswith('.tsp')]
        return tsp_files
    return []


def download_tsplib_instance(instance_name):
    """Download a TSP instance from the TSPLIB library.
    
    Args:
        instance_name: Name of the instance to download (e.g., 'a280', 'att48')
        
    Returns:
        Path to the downloaded file
    """
    import urllib.request
    
    # TSPLIB base URL
    tsplib_url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
    instance_filename = f"{instance_name}.tsp"
    instance_url = f"{tsplib_url}{instance_filename}.gz"
    
    local_path = os.path.join(DEFAULT_TSP_DIR, instance_filename)
    
    # Don't download if already exists
    if os.path.exists(local_path):
        print(f"Instance {instance_name} already exists at {local_path}")
        return local_path
    
    # Download and extract the gzipped file
    import gzip
    import shutil
    
    try:
        print(f"Downloading {instance_name} from {instance_url}...")
        # Download to a temporary gzipped file
        temp_gz_path = os.path.join(DEFAULT_TSP_DIR, f"{instance_filename}.gz")
        urllib.request.urlretrieve(instance_url, temp_gz_path)
        
        # Extract the gzipped file
        with gzip.open(temp_gz_path, 'rb') as f_in:
            with open(local_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove the temporary gzipped file
        os.remove(temp_gz_path)
        
        print(f"Successfully downloaded {instance_name} to {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading {instance_name}: {e}")
        return None

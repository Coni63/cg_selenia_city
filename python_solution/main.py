import sys
import math

import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist

class TubePlacer:
    def __init__(self, points):
        """
        Initialize the tube placer with a set of points.
        
        Args:
            points: numpy array of shape (n, 2) containing point coordinates
        """
        if len(points) == 2:
            self.points = np.array(points)
            self.existing_tubes = set()
            self.edges = {(0, 1)}
        else:
            self.points = np.array(points)
            self.tri = Delaunay(points)
            self.existing_tubes = set()  # Store pairs of point indices that have tubes
            
            # Create initial graph of possible connections
            self.edges = set()
            for simplex in self.tri.simplices:
                n = len(simplex)
                for i in range(n):
                    for j in range(i + 1, n):
                        self.edges.add(tuple(sorted([simplex[i], simplex[j]])))
    
    def line_segments_intersect(self, p1, p2, p3, p4):
        """
        Check if line segments (p1,p2) and (p3,p4) intersect.
        Points are given as coordinate pairs.
        """
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def would_intersect(self, point_indices):
        """
        Check if a new tube would intersect with existing tubes.
        
        Args:
            point_indices: tuple of (start_idx, end_idx) for the new tube
        """
        new_p1 = self.points[point_indices[0]]
        new_p2 = self.points[point_indices[1]]
        
        for tube in self.existing_tubes:
            if set(tube) & set(point_indices):  # Skip if tubes share an endpoint
                continue
            
            p1 = self.points[tube[0]]
            p2 = self.points[tube[1]]
            
            if self.line_segments_intersect(new_p1, new_p2, p1, p2):
                return True
        
        return False
    
    def get_valid_edges(self):
        """
        Return all valid edges that don't intersect with existing tubes.
        """
        valid_edges = set()
        for edge in self.edges:
            if edge not in self.existing_tubes and not self.would_intersect(edge):
                valid_edges.add(edge)
        return valid_edges
    
    def add_tube(self, start_idx, end_idx):
        """
        Add a new tube between two points.
        
        Args:
            start_idx: index of starting point
            end_idx: index of ending point
        """
        edge = tuple(sorted([start_idx, end_idx]))
        if edge in self.edges and not self.would_intersect(edge):
            self.existing_tubes.add(edge)
            return True
        return False
    
    def find_shortest_path(self, start_idx, end_idx):
        """
        Find shortest path between two points using only valid edges.
        
        Args:
            start_idx: index of starting point
            end_idx: index of ending point
        """
        valid_edges = self.get_valid_edges()
        n_points = len(self.points)
        
        # Create adjacency matrix
        adj_matrix = np.zeros((n_points, n_points))
        for edge in valid_edges:
            i, j = edge
            dist = np.linalg.norm(self.points[i] - self.points[j])
            adj_matrix[i, j] = dist
            adj_matrix[j, i] = dist
        
        # Find shortest path
        graph = csr_matrix(adj_matrix)
        dist_matrix, predecessors = shortest_path(graph, 
                                               directed=False, 
                                               indices=[start_idx],
                                               return_predecessors=True)
        
        # Reconstruct path
        path = []
        if predecessors[0, end_idx] != -9999:  # Check if path exists
            current = end_idx
            while current != start_idx:
                path.append(current)
                current = predecessors[0, current]
            path.append(start_idx)
            path.reverse()
            
        return path if path else None

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.
class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Pod:
    def __init__(self, _id: int, path: list[int]):
        self.id = _id
        self.path = path

class Building(Position):
    def __init__(self, _id: int, _type: int, x: int, y: int, crew: list[int]):
        super(Building, self).__init__(x, y)
        self.id = _id
        self.type = _type
        self.crew = [0 for i in range(21)]

        for i in crew:
            self.crew[i] += 1

    def __repr__(self):
        return f"Building type {self.type} #{self.id} ({self.x}, {self.y}) => {self.crew}"

all_buildings = []
positions = []
# game loop
while True:
    resources = int(input())
    print(f"{resources} resources", file=sys.stderr)
    
    all_routes = []
    present = set()
    num_travel_routes = int(input())
    print(f"{num_travel_routes} routes", file=sys.stderr)
    for i in range(num_travel_routes):
        building_id_1, building_id_2, capacity = [int(j) for j in input().split()]
        present.add(tuple(sorted([building_id_1, building_id_2])))
        print(f"Route {building_id_1} -> {building_id_2} ({capacity})", file=sys.stderr)
    
    all_pods = []
    num_pods = int(input())
    print(f"{num_pods} pods", file=sys.stderr)
    for i in range(num_pods):
        id_, num_nodes, *path = [int(x) for x in input().split()]
        all_pods.append(Pod(id_, path))
        print(id_, num_nodes, path, file=sys.stderr)
    
    num_new_buildings = int(input())
    print(f"{num_new_buildings} buildings", file=sys.stderr)
    for i in range(num_new_buildings):
        s = input().split()
        # print(s, file=sys.stderr)
        if len(s) == 4:
            type_, id_, x, y = [int(x) for x in s]
            crew = []
        else:
            type_, id_, x, y, num_crew, *crew = [int(x) for x in s]
        all_buildings.append(Building(type_, id_, x, y, crew))
        positions.append( [float(x), float(y)])

    placer = TubePlacer(positions)
    print(placer.edges, file=sys.stderr)

    dist_cost = cdist(positions, positions, metric='euclidean') / 10
    print(dist_cost, file=sys.stderr)

    missing_edges = placer.edges - present

    s = ["WAIT"]
    for i, (a, b) in enumerate(missing_edges):
        cost = math.floor(dist_cost[a, b]) + 1000

        if cost < resources:
            resources -= cost
            s.append(f"TUBE {a} {b}")
            s.append(f"POD {num_pods + i} {a} {b} {a}")



    print(";".join(s))
    # # Add some tubes
    # placer.add_tube(0, 1)
    # placer.add_tube(1, 2)
    # placer.add_tube(0, 2)
    
    # # Find path between points
    # path = placer.find_shortest_path(0, 2)
    # print(f"Shortest path from point 0 to 2: {path}")
    
    # # Get remaining valid edges
    # valid_edges = placer.get_valid_edges()
    # print(f"Valid edges remaining: {valid_edges}")

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    # TUBE | UPGRADE | TELEPORT | POD | DESTROY | WAIT
    # print("TUBE 0 1;TUBE 0 2;POD 42 0 1 0 2 0 1 0 2")

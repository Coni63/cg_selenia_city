import sys
import math
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from queue import PriorityQueue
from dataclasses import dataclass

class TubePlacer:
    def __init__(self, points):
        """
        Initialize the tube placer with a set of points.
        
        Args:
            points: numpy array of shape (n, 2) containing point coordinates
        """
        self.points = np.array(points)


        temp_d = cdist(self.points, self.points, metric='euclidean') / 10
        self.dist_cost = np.zeros_like(temp_d) - 1
        if len(points) == 2:
            self.existing_tubes = set()
            self.edges = {(0, 1)}
        else:
            self.existing_tubes = set()  # Store pairs of point indices that have tubes
            # Create initial graph of possible connections
            self.edges = set()
            for simplex in Delaunay(points).simplices:
                n = len(simplex)
                for i in range(n):
                    for j in range(i + 1, n):
                        self.edges.add(tuple(sorted([simplex[i], simplex[j]])))

        print(f"{len(self.points)} vs {len(self.edges)}", file=sys.stderr)

        for a, b in self.edges:
            self.dist_cost[a, b] = temp_d[a, b]
            self.dist_cost[b, a] = temp_d[b, a]
    
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
            self.dist_cost[start_idx, end_idx] = 0.0
            self.dist_cost[end_idx, start_idx] = 0.0
            return True
        return False
    
    def find_shortest_path(self, start_idx: int, end_idx: int) -> tuple[list[int], float]:
        """
        Find shortest path between two points using A* algorithm.
        
        Args:
            start_idx: index of starting point
            end_idx: index of ending point

        Returns:
            tuple[list[int], float]: (sequence of node indices, total distance)
        """
        # Initialize data structures
        frontier = PriorityQueue()
        frontier.put((0, start_idx))
        came_from = {start_idx: None}
        cost_so_far = {start_idx: 0}
        
        def get_neighbors(node_idx):
            """Get valid neighbors of a node."""
            neighbors = []
            for neighbor_idx, dist in enumerate(self.dist_cost[node_idx]):
                if dist >= 0:  # Valid edge
                    neighbors.append((neighbor_idx, dist))
            return neighbors
        
        def heuristic(node_idx):
            """
            Heuristic function for A*.
            Using 0 makes this equivalent to Dijkstra's algorithm.
            """
            return 0
        
        # A* algorithm
        while not frontier.empty():
            current_priority, current = frontier.get()
            
            # Exit if we reached the goal
            if current == end_idx:
                break
                
            # Check all neighbors
            for next_node, dist in get_neighbors(current):
                new_cost = cost_so_far[current] + dist
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(next_node)
                    frontier.put((priority, next_node))
                    came_from[next_node] = current
        
        # Reconstruct path
        if end_idx not in came_from:  # No path found
            return [], float('inf')
            
        path = []
        current = end_idx
        
        while current is not None:
            path.append(current)
            current = came_from[current]
        
        path.reverse()
        
        return path, cost_so_far[end_idx]



@dataclass
class Position:
    x: int
    y: int

@dataclass
class Pod:
    id: int  # must be > 150
    path: list[int]

@dataclass
class Tube:
    source: int
    target: int
    capacity: int

    def is_teleporter(self):
        return self.capacity == 0

@dataclass
class Building:
    id: int  # 0 - 150
    x: int
    y: int
    type: int
    crew: list[int]

    def is_landing_pad(self):
        return self.type == 0
    
    def has_crew_of_type(self, crew_type):
        return self.type > 0 and self.crew[crew_type] > 0
    
    def is_type_of(self, building_type):
        return self.type == building_type


def get_cost_matrix(buildings: list[Building], existing_tubes: list[Tube]) -> np.ndarray:
    """
    Create a cost matrix for the buildings based on their coordinates.
    
    Args:
        buildings: list of Building objects
        existing_tubes: list of Tube objects representing existing tubes

    Returns:
        numpy array: cost matrix
    """
    coords = np.array([[b.x, b.y] for b in buildings])

    # Calculate the pairwise distances between buildings
    temp_d = cdist(coords, coords, metric='euclidean') / 10

    # Initialize the resulting matrix with -1 to indicate no connection
    dist_cost = np.full(temp_d.shape, np.inf)
    
    # Fill the matrix with distances only for Delaunay edges
    if len(coords) == 2:
        dist_cost[0, 1] = temp_d[0, 1]
        dist_cost[1, 0] = temp_d[1, 0]
    else:
        for simplex in Delaunay(coords).simplices:
            n = len(simplex)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = simplex[i], simplex[j]
                    dist_cost[a, b] = temp_d[a, b]
                    dist_cost[b, a] = temp_d[b, a]

    # Set distances for existing tubes to 0
    for tube in existing_tubes:
        a, b = tube.source, tube.target
        dist_cost[a, b] = 0.0
        dist_cost[b, a] = 0.0

    return dist_cost

def find_shortest_path(source: int, target: int, cost_matrix: np.ndarray) -> tuple[list[int], float]:
    distances, predecessors = dijkstra(csgraph=cost_matrix, directed=False, indices=source, return_predecessors=True)
    path = []
    i = target
    while i != -9999:
        path.append(i)
        i = predecessors[i]
    path.append(source)
    return path[::-1], distances[target]


def render(cost_matrix: np.ndarray, buildings: list[Building], existing_tubes: list[Tube]):
    with open("render.log", "a") as f:
        try:
            import matplotlib.pyplot as plt

            colors = [
                (0.0, 0.0, 0.0),
                (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                (1.0, 0.4980392156862745, 0.054901960784313725),
                (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
                (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
                (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
                (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
                (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
                (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
                (1.0, 0.7333333333333333, 0.47058823529411764),
                (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
                (1.0, 0.596078431372549, 0.5882352941176471),
                (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
                (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
                (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
                (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
                (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
                (0.6196078431372549, 0.8549019607843137, 0.8980392156862745)
            ]

            f.write(f"Number of buildings: {len(buildings)}\n")
            for i, building in enumerate(buildings):
                plt.scatter(building.x, building.y, label=f"Building {building.id}", s=100, color=colors[building.type])

            for tube in existing_tubes:
                if tube.is_teleporter():
                    plt.plot([buildings[tube.source].x, buildings[tube.target].x],
                             [buildings[tube.source].y, buildings[tube.target].y], color="red", linewidth=2)
                else:
                    plt.plot([buildings[tube.source].x, buildings[tube.target].x],
                             [buildings[tube.source].y, buildings[tube.target].y], color="blue", linewidth=2)
                    
            for i in range(len(cost_matrix)):
                for j in range(i + 1, len(cost_matrix)):
                    cost = cost_matrix[i][j]
                    if 0 < cost < np.inf:
                        plt.plot([buildings[i].x, buildings[j].x],
                                [buildings[i].y, buildings[j].y],
                                linestyle='dashed', color="green", linewidth=1)
            
            plt.savefig("buildings.png")
            plt.show()
        except Exception as e:
            f.write(f"Error in rendering: {e}\n")


all_buildings: list[Building] = []
# game loop
while True:
    resources = int(input())
    # print(f"{resources} resources", file=sys.stderr)
    
    all_present_routes = []
    num_travel_routes = int(input())
    for i in range(num_travel_routes):
        building_id_1, building_id_2, capacity = [int(j) for j in input().split()]
        all_present_routes.append(Tube(source=building_id_1, target=building_id_2, capacity=capacity))
    
    all_present_pods = []
    num_pods = int(input())
    for i in range(num_pods):
        id_, num_nodes, *path = [int(x) for x in input().split()]
        all_present_pods.append(Pod(id_, path))
    
    num_new_buildings = int(input())
    for i in range(num_new_buildings):
        s = input().split()
        if len(s) == 4:
            id_, type_, x, y = [int(x) for x in s]
            grouped_crew = [0 for _ in range(21)]
        else:
            id_, type_, x, y, num_crew, *crew = [int(x) for x in s]
            grouped_crew = [0 for _ in range(21)]
            for crew_id in crew:
                grouped_crew[crew_id] += 1
        all_buildings.append(Building(id_, x, y, type_, grouped_crew))
    all_buildings.sort(key=lambda b: b.id)

    cost_matrix = get_cost_matrix(all_buildings, all_present_routes)
    print(cost_matrix, file=sys.stderr)

    render(cost_matrix, all_buildings, all_present_routes)

    tic = time.time()

    s = ["WAIT"]
    # for base in all_buildings:
    #     if base.is_landing_pad():
    #         if time.time() - tic > 0.4:
    #             break

    #         print(f"base {base.id}", file=sys.stderr)
    #         for i in range(1, 21):
    #             if base.has_crew_of_type(i) > 0:
    #                 best_dist, best_path = 999999999, []
    #                 for building in all_buildings:
    #                     if building.is_type_of(i):
    #                         path, total_cost = find_shortest_path(base.id, building.id, cost_matrix)
    #                         print(f"Shortest path from point {base.id} to {building.id}: {path}", file=sys.stderr)
    #                         # print(f"Total Cost: {total_cost}", file=sys.stderr)

    #                         if total_cost == np.inf:
    #                             continue

    #                         cost = math.floor(total_cost) + 1000
    #                         if cost < best_dist:
    #                             best_dist = cost
    #                             best_path = path
                            

    #                 if best_dist < resources:
    #                     resources -= best_dist
    #                     for a, b in zip(best_path[:-1], best_path[1:]):
    #                         if cost_matrix[a, b] > 0:
    #                             s.append(f"TUBE {a} {b}")
    #                             cost_matrix[a, b] = 0.0
    #                             cost_matrix[b, a] = 0.0
    #                     seq = " ".join(map(str, best_path + best_path[:-1][::-1]))
    #                     pod_index = 200 + len(all_present_pods)
    #                     s.append(f"POD {pod_index} {seq}")

    #                 if resources <= 1000:
    #                     break

    # TUBE | UPGRADE | TELEPORT | POD | DESTROY | WAIT
    # print("TUBE 0 1;TUBE 0 2;POD 42 0 1 0 2 0 1 0 2")

    print(";".join(s))
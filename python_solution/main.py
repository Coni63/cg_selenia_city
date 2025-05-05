import sys
import math
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from dataclasses import dataclass

f = open("render.log", "w")
f2 = open("out.log", "w")

def debug_print(*args, **kwargs):
    print(*args, file=f, flush=True)


@dataclass
class Pod:
    id: int  # must be > 150
    path: list[int]

    def __str__(self):
        path_str = " ".join(map(str, self.path))
        return f"POD {self.id} {path_str}"

@dataclass
class Tube:
    source: int
    target: int
    capacity: int

    def is_teleporter(self):
        return self.capacity == 0
    
    def __str__(self):
        return f"TUBE {self.source} {self.target}"

@dataclass
class Building:
    id: int  # 0 - 150
    x: int
    y: int
    type: int
    crew: list[int]

    def is_landing_pad(self):
        return self.is_type_of(0)
    
    def has_crew_of_type(self, crew_type: int) -> bool:
        return self.is_landing_pad() and self.crew[crew_type] > 0
    
    def is_type_of(self, building_type):
        return self.type == building_type
    
    def __repr__(self):
        crew_str = " ".join(map(str, self.crew))
        return f"Building {self.id} (type {self.type}) has grouped_crew: {crew_str}"


def get_distance_matrix(buildings: list[Building]) -> np.ndarray:
    """
    Create a distance matrix for the buildings based on their coordinates.
    
    Args:
        buildings: list of Building objects

    Returns:
        numpy array: distance matrix
    """
    coords = np.array([[b.x, b.y] for b in buildings])

    # Calculate the pairwise distances between buildings
    temp_d = cdist(coords, coords, metric='euclidean')

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

    return dist_cost

def get_cost_matrix(distance_matrix: np.ndarray, existing_tubes: list[Tube]) -> np.ndarray:
    """
    Create a cost matrix for the buildings based on their coordinates.
    
    Args:
        distance_matrix: numpy array representing the distance matrix
        existing_tubes: list of Tube objects representing existing tubes

    Returns:
        numpy array: cost matrix
    """
    dist_cost = np.floor(distance_matrix.copy() * 10)

    # Set distances for existing tubes to 0.001 (do not set 0 otherwise it's like closed path)
    for tube in existing_tubes:
        a, b = tube.source, tube.target
        dist_cost[a, b] = 0.001
        dist_cost[b, a] = 0.001

    return dist_cost

def find_shortest_path(distances: np.ndarray, predecessors: np.ndarray, source: int, target: int) -> tuple[list[int], float]:
    path = []
    i = target
    while i != -9999 and i != source:
        path.append(int(i))
        i = predecessors[i]
    if i == -9999:
        return [], float('inf')  # unreachable
    path.append(source)
    return path[::-1], distances[target]


def render(cost_matrix: np.ndarray, buildings: list[Building], existing_tubes: list[Tube]):
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

        debug_print(f"Number of buildings: {len(buildings)}\n")
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
        # plt.show()
    except Exception as e:
        debug_print(f"Error in rendering: {e}")


all_buildings: list[Building] = []
turn = 1
# game loop
while True:
    resources = int(input())
    debug_print(f"\n\n#### Turn: {turn} ####\n\n{resources} resources")
    turn += 1
    
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
            # Target base, not a landing pad
            type_, id_, x, y = [int(x) for x in s]
            grouped_crew = []
        else:
            # Landing pad with crew
            type_, id_, x, y, num_crew, *crew = [int(x) for x in s]
            grouped_crew = [0 for _ in range(21)]
            for crew_id in crew:
                grouped_crew[crew_id] += 1
        building = Building(id_, x, y, type_, grouped_crew)
        debug_print(building)
        all_buildings.append(building)
    all_buildings.sort(key=lambda b: b.id)

    distance_matrix = get_distance_matrix(all_buildings)
    cost_matrix = get_cost_matrix(distance_matrix, all_present_routes)
    debug_print(f"Distance:\n{distance_matrix}")
    debug_print(f"Cost:\n{cost_matrix}")

    # render(cost_matrix, all_buildings, all_present_routes)

    tic = time.time()

    s = ["WAIT"]
    for base in all_buildings:
        if not base.is_landing_pad():
            continue

        distances, predecessors = dijkstra(csgraph=cost_matrix, directed=False, indices=base.id, return_predecessors=True)
        
        for i in range(1, 21):
            if not base.has_crew_of_type(i):
                continue

            best_dist, best_path = np.inf, []
            for building in all_buildings:
                if building.is_type_of(i):
                    path, total_cost = find_shortest_path(distances, predecessors, base.id, building.id)
                    # debug_print(f"Shortest path from base {base.id} to base {building.id}: {path}")
                    # debug_print(f"Total Cost: {total_cost}")

                    if total_cost == np.inf or total_cost < 1:
                        continue

                    if total_cost < best_dist:
                        best_dist = total_cost
                        best_path = path
            # debug_print(f"Shortest path for entity type {i}: {best_path} ({best_dist})")

            debug_print(f"Shortest path from base {base.id} to base {i}: {path} ({best_dist}, {resources})")
            num_tubes = len(best_path) - 1
            if resources > best_dist + num_tubes * 1000:
                for a, b in zip(best_path[:-1], best_path[1:]):
                    if cost_matrix[a][b] > 0.0015 and resources >= cost_matrix[a][b] + 1000:
                        resources -= (cost_matrix[a][b] + 1000)
                        tube = Tube(a, b, 1)
                        s.append(str(tube))
                        cost_matrix[a][b] = 0.001
                        cost_matrix[b][a] = 0.001
                        pod_index = 200 + len(all_present_pods)
                        pod = Pod(pod_index, [a, b, a])
                        all_present_pods.append(pod)
                        s.append(str(pod))

                    if resources < 1000:
                        break
            elif resources > best_dist + 1000:
                resources -= (best_dist + 1000)
                for a, b in zip(best_path[:-1], best_path[1:]):
                    if cost_matrix[a, b] > 0:
                        tube = Tube(a, b, 1)
                        s.append(str(tube))
                        cost_matrix[a, b] = 0.0
                        cost_matrix[b, a] = 0.0
                seq = " ".join(map(str, best_path + best_path[:-1][::-1]))
                pod_index = 200 + len(all_present_pods)
                pod = Pod(pod_index, best_path + best_path[:-1][::-1])
                all_present_pods.append(pod)
                s.append(str(pod))

            debug_print(f" Remaining: {resources}")

            if resources < 1000:
                break



    # TUBE | UPGRADE | TELEPORT | POD | DESTROY | WAIT
    # print("TUBE 0 1;TUBE 0 2;POD 42 0 1 0 2 0 1 0 2")

    print(";".join(s), file=f2, flush=True)
    print(";".join(s))
    debug_print(f"Time taken: {time.time() - tic:.2f} seconds")
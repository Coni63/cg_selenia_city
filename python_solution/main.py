import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.
f = open("test.txt", "w")

sys.stderr = f

# game loop
while True:
    resources = int(input())
    print(f"{resources} resources", file=sys.stderr)
    num_travel_routes = int(input())
    print(f"{num_travel_routes} routes", file=sys.stderr)
    for i in range(num_travel_routes):
        building_id_1, building_id_2, capacity = [int(j) for j in input().split()]
        print(f"Route {building_id_1} -> {building_id_2} ({capacity})", file=sys.stderr)
    num_pods = int(input())
    print(f"{num_pods} pods", file=sys.stderr)
    for i in range(num_pods):
        pod_properties = input()
        print(f"{pod_properties}", file=sys.stderr)
    num_new_buildings = int(input())
    print(f"{num_new_buildings} buildings", file=sys.stderr)
    for i in range(num_new_buildings):
        building_properties = input()
        print(building_properties, file=sys.stderr)

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    # TUBE | UPGRADE | TELEPORT | POD | DESTROY | WAIT
    print("TUBE 0 1;TUBE 0 2;POD 42 0 1 0 2 0 1 0 2")

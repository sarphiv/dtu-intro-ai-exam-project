import numpy as np
import math

map_path = "maps.npy"
map_global_scale = 0.9
maps = np.load(map_path, allow_pickle=True)
map_amount = maps.shape[0]

def get_min_max(map):
    #Get x,y-coordinates
    map_coords = map.reshape((-1, 2))

    #Get minimum x,y coordinates
    return np.min(map_coords, axis=0), np.max(map_coords, axis=0)


def get_map(id, direction, width, height):
    #Retrieve map from cache
    map = maps[id][:][::-1 if direction else 1]
    map = np.array([map[0], map[1]])

    #Get smallest dimension of playground
    if width < height:
        map_coords_index = 0
        map_scaler = width
    else:
        map_coords_index = 1
        map_scaler = height


    #Get map limits
    (map_min, map_max) = get_min_max(map)
    min = map_min[map_coords_index]
    max = map_max[map_coords_index]
    
    #Normalize map
    nrm_map = (map - min) / (max - min)
    #Scale map
    scaled_map = nrm_map * map_scaler * map_global_scale
    
    #Get scaled map limits
    (map_min, map_max) = get_min_max(scaled_map)
    
    #Calculate centering offset
    map_offset = np.array([width, height]) / 2
    map_offset += -map_min - (map_max - map_min) / 2
    
    #Offset map
    scaled_map += map_offset.reshape(1, 1, 2)

    #Return map
    return scaled_map

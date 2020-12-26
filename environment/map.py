import numpy as np

def create_empty_map(width, height):
    (w, h) = (width, height)
    edge_depth = max(w, h) / 4
    edge_right = [(0, 0), (-edge_depth, 0), (-edge_depth, h), (0, h)]
    edge_left = [(w, 0), (w+edge_depth, 0), (w+edge_depth, h), (w, h)]
    edge_top = [(0, 0), (0, -edge_depth), (w, -edge_depth), (w, 0)]
    edge_bottom = [(0, h), (0, h+edge_depth), (w, h+edge_depth), (w, h)]
    
    return [edge_right, edge_left, edge_top, edge_bottom]


def create_middle_obstacle(map_width, map_height):
    x_scale = 0.05
    y_scale = 0.05
    
    width = x_scale * map_width
    height = y_scale * map_height
    
    top_left     = np.array([(map_width - width) / 2, (map_height - height) / 2])
    bottom_right = top_left + np.array([width, height])
    
    (x1, y1) = top_left
    (x2, y2) = bottom_right

    return [(x1, y1), (x1+width/2, y1-height/2), (x2, y1), (x2, y2),(x2-width/2, y2+height/2), (x1, y2)]
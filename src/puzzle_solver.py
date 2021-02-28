from time import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import random as rng
from itertools import combinations
from scipy.ndimage import interpolation
from .cfg import CFG

class PuzzleSolver:
    
    def __init__(self, puzzles, edge_distance_dict, verbose=True, puzzles_size=(300, 300), distance_maximum=1.0, min_puzzle_show=16):
        self.puzzles = puzzles
        self.edge_distance_dict = edge_distance_dict
        self.edges = set(e for p in puzzles for e in p.edges)
        self.edges_to_connect = set([e for e in self.edges if e.edge_type != 'straight'])
        
        self.verbose = verbose
        self.puzzles_size = puzzles_size
        self.inner_puzzles_num = len([p for p in puzzles if all(e.edge_type != 'straight' for e in p.edges)])
        self.outer_puzzles_num = len([p for p in puzzles if any(e.edge_type == 'straight' for e in p.edges)])
        
        self.count_rows_and_cols()
        
        self.results = []
        
        self.back_num = 0
        self.depth = 0
        self.distance_maximum = distance_maximum
        self.min_puzzle_show = min_puzzle_show
            
        if self.verbose:
            print(f'inner_puzzles_num = {self.inner_puzzles_num}')
            print(f'outer_puzzles_num = {self.outer_puzzles_num}')

            print(f'rows_num {self.rows_num}')
            print(f'cols_num {self.cols_num}')
        
    def count_rows_and_cols(self):
        
        n = self.inner_puzzles_num 
        m = int((self.outer_puzzles_num - 4) / 2)
        self.rows_num = int((m + np.sqrt((m ** 2) - 4 * n)) / 2) + 2
        self.cols_num = int((m - np.sqrt((m ** 2) - 4 * n)) / 2) + 2
    
    def get_first_puzzle_edges(self, corner_puzzle):
        """ Returns edges of the given corner puzzle in a clockwise order:
        (upper, right, bottom, left), where upper and
        left edges are straight. 
        """
        e1, e2, e3, e4 = corner_puzzle.edges
        
        if e1.edge_type == 'straight' and e4.edge_type == 'straight':  # correct order
            return e1, e2, e3, e4
        else:
            straight_edge = next(e for e in [e1, e2, e3, e4] if e.edge_type == 'straight')
            straight_edge_idx = [e1, e2, e3, e4].index(straight_edge)
            
            return np.roll([e1, e2, e3, e4], 3 - straight_edge_idx)
    
    def solve(self):
        corners = [p for p in self.puzzles if p.is_corner]
        for corner in corners:
            connections = [[self.get_first_puzzle_edges(corner)]]
            connected_edges = set()
            connected_edges.update(corner.edges)
            res = self._backtrack(connections, connected_edges=connected_edges)
            
            if res:
                return res
        
    def draw_img(self, img):
        if self.rows_num > self.cols_num:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        gray = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY)
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
        _, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        img = img[y:y+h,x:x+w]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        plt.show()
    
    def draw_solution(self, connections):
        """
            connections - list of lists of size num_rows x num_cols x 4
            Each element of a list is tuple of 4 edge references:
                (upper edge, right edge, bottom edge, left edge)
        """
        edge_distance_dict = self.edge_distance_dict
        puzzles_size = self.puzzles_size 
        puzzles_image = np.zeros(puzzles_size + (3,))
        edge_fixed_points = {}
        
        top_distances, left_distances = [], []
        
        for row_idx in range(self.rows_num):
            for col_idx in range(self.cols_num):
                
                if row_idx >= len(connections) or col_idx >= len(connections[row_idx]):
                    continue
                    
                u, r, b, l = connections[row_idx][col_idx]
                puzzle = u.puzzle
                puzzle_image = puzzle.puzzle_image_no_bg
                top_neighbour = None
                left_neighbour = None
                right_top_corner = None
                
                if col_idx > 0:
                    left_neighbour = connections[row_idx][col_idx - 1][1]
                    
                    left_top_corner = edge_fixed_points[left_neighbour][0]
                    left_bottom_corner = edge_fixed_points[left_neighbour][-1]
                elif row_idx > 0:
                    top_neighbour = connections[row_idx-1][col_idx][2]
                    
                    left_top_corner = edge_fixed_points[top_neighbour][-1]
                    left_bottom_corner = (left_top_corner[0], left_top_corner[1] + l.length)
                else:
                    left_top_corner = (0, 0)
                    left_bottom_corner = (left_top_corner[0], left_top_corner[1] + l.length)
                
                if row_idx > 0:
                    top_neighbour = connections[row_idx - 1][col_idx][2]
                    right_top_corner = edge_fixed_points[top_neighbour][1]
                #else:
                    #right_top_corner = (left_top_corner[0] + l.length, left_top_corner[1])
                
                if right_top_corner is not None:
                    v1 = np.array(right_top_corner) - np.array(left_bottom_corner)
                    v2 = np.array(u.points[-1]) - np.array(l.points[0])
                else:
                    v1 = np.array(left_top_corner) - np.array(left_bottom_corner)
                    v2 = np.array(l.points[-1]) - np.array(l.points[0])
                
                dot = v1[0] * v2[0] + v1[1] * v2[1]      # dot product between [x1, y1] and [x2, y2]
                det = v1[0] * v2[1] - v1[1] * v2[0]
                alpha = np.arctan2(det, dot)
                
                rows, cols, _ = puzzle_image.shape
                warp_mat = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 180*alpha/np.pi, 1)
                
                for e in puzzle.edges:
                    edge_fixed_points[e] = e.points.dot(warp_mat[:, :2].T) + warp_mat[:, 2]
                
                bias = np.array(left_bottom_corner) - np.array(edge_fixed_points[l][0])
                warp_mat[:, 2] = warp_mat[:, 2] + bias
                
                for e in puzzle.edges:
                    edge_fixed_points[e] = e.points.dot(warp_mat[:, :2].T) + warp_mat[:, 2]

                warp_puzzle_image = cv2.warpAffine(puzzle_image, warp_mat, puzzles_size)
                
                puzzles_image = cv2.addWeighted(puzzles_image.astype(int), 1, warp_puzzle_image.astype(int), 1, 0.0)
                puzzles_image[puzzles_image > 255] = 255
                
                if top_neighbour and edge_distance_dict:
                    dist = edge_distance_dict[u, top_neighbour]
                    top_distances.append(dist)
                else:
                    top_distances.append(None)
                if left_neighbour and edge_distance_dict:
                    dist = edge_distance_dict[l, left_neighbour]
                    left_distances.append(dist)
                else:
                    left_distances.append(None)

        if self.verbose:
            print('top distances',top_distances)
            print('left distances',left_distances)
            print('back num',self.back_num)

        self.draw_img(puzzles_image)
              
    def _backtrack(self, connections, connected_edges, depth=0):
        edge_distance_dict = self.edge_distance_dict

        if self.min_puzzle_show and depth > self.min_puzzle_show:
            self.draw_solution(connections)
            
        if len(connected_edges) == len(self.edges):
            t0 = time()
            print('Number of backtracks: ', self.back_num)
            self.draw_solution(connections)
            return True #connections
        
        new_connections = []
        for row in connections:
            new_connections.append([])
            for col in row:
                new_connections[-1].append(col)

        last_column = False
        if len(new_connections[-1]) == self.cols_num - 1:
            last_column = True

        if len(new_connections[-1]) == self.cols_num:
            new_connections.append([])
            left_neighbour_edge = None
        else:
            left_neighbour_edge = connections[-1][-1][1]
        
        if len(new_connections) == 1:  # first row
            top_neighbour_edge = None
        else:
            col_idx = len(new_connections[-1])
            top_neighbour_edge = new_connections[-2][col_idx][2]
        
        def distance_sum(edge):
            distances = []
            if left_neighbour_edge is not None:
                dist = edge_distance_dict[edge, left_neighbour_edge]
                distances.append(dist)
                #dist += np.abs(len(edge.points) - len(left_neighbour_edge.points))
            if top_neighbour_edge is not None:
                dist = edge_distance_dict[edge.next_edge, top_neighbour_edge]
                distances.append(dist)
                #dist += np.abs(len(edge.next_edge.points) - len(top_neighbour_edge.points))
            
            return np.mean(distances)
        
        edges_to_check = list(self.edges.difference(connected_edges))
        edges_to_check = sorted(edges_to_check, key=distance_sum) 
        
        for e in edges_to_check:
            if distance_sum(e) > self.distance_maximum: # for color + affine distance
                self.back_num += 1
                return False

            if left_neighbour_edge and not left_neighbour_edge.can_be_connected_with(e):
                continue
            if top_neighbour_edge and not top_neighbour_edge.can_be_connected_with(e.next_edge):
                continue
            
            if last_column and not e.next_edge.next_edge.edge_type == 'straight':
                continue
            
            l, u, r, b = e, e.next_edge, e.next_edge.next_edge, e.next_edge.next_edge.next_edge
            
            new_connections[-1].append((u, r, b, l))
            
            new_connected_edges = set(connected_edges)
            new_connected_edges.update([u, r, b, l])
            
            res = self._backtrack(new_connections, new_connected_edges, depth+1)
            
            if res:
                #self.draw_solution(new_connections)
                return res
            
            new_connections[-1].pop()
        
        
        return False
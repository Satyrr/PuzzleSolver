import numpy as np
import cv2
import matplotlib.pyplot as plt
from .cfg import CFG

import random as rng
from itertools import combinations
from scipy.ndimage import interpolation

class Edge:
    @property
    def edge_type(self):
        if len(self.defects) == 1:
            return 'hole'
        elif len(self.defects) == 2:
            return 'header'
        else:
            return 'straight'
    
    @property
    def next_edge(self):
        own_puzzle_edges = self.puzzle.edges
        self_index = own_puzzle_edges.index(self)
        
        return self.puzzle.edges[(self_index + 1) % 4]
        
        
    @property
    def prev_edge(self):
        own_puzzle_edges = self.puzzle.edges
        self_index = own_puzzle_edges.index(self)
        
        return self.puzzle.edges[(self_index - 1) % 4]
    
    @property
    def length(self):
        return np.linalg.norm(self.points[0] - self.points[-1])
        
    def __init__(self, puzzle, edge_points, edge_defects):
        self.puzzle = puzzle
        self.points = edge_points
        self.defects = edge_defects
    
    def get_color_array(self, thickness):
        puzzle_image = self.puzzle.puzzle_image.copy()
        edge_mask = np.zeros((puzzle_image.shape[0], puzzle_image.shape[1], 3), dtype=np.uint8)
        
        cv2.polylines(edge_mask, 
          [np.array(self.points)], 
          isClosed = False,
          color = (255,255,255),
          thickness = thickness)
        
        puzzle_image[edge_mask != 255] = 0
        puzzle_image[np.all(puzzle_image >= 255, axis=2)] = 0
        
        return puzzle_image
    
    def draw_edge(self, width=3):
        edge_array = self.get_color_array(width)
    
        plt.figure(figsize=(10, 10))
        plt.title('edge')
        plt.imshow(edge_array)
        plt.show()
        
    def can_be_connected_with(self, other_edge):
        if self.puzzle == other_edge.puzzle:
            return False
        
        if 'straight' in [self.edge_type, other_edge.edge_type]:
            return False
        
        if self.edge_type == other_edge.edge_type:
            return False
        
        own_puzzle_edges = self.puzzle.edges
        other_puzzle_edges = other_edge.puzzle.edges
            
        self_index = own_puzzle_edges.index(self)
        self_left_neighbour = own_puzzle_edges[self_index-1]
        self_right_neighbour = own_puzzle_edges[(self_index+1) % 4] 

        other_index = other_puzzle_edges.index(other_edge)
        other_left_neighbour = other_puzzle_edges[other_index-1]
        other_right_neighbour = other_puzzle_edges[(other_index+1) % 4] 
        
        
        if sum(1 if e.edge_type == 'straight' else 0 for e in [self_left_neighbour, other_right_neighbour]) == 1:
            return False
        
        if sum(1 if e.edge_type == 'straight' else 0 for e in [self_right_neighbour, other_left_neighbour]) == 1:
            return False
    
        return True

        
    def color_distance(self, other_edge):
        
        self_edge_mat = (self.get_color_array(3) > 0).any(axis=2) 
        other_edge_mat = (other_edge.get_color_array(3) > 0).any(axis=2) 
        
        self_mean_color = self.get_color_array(3)[self_edge_mat].mean(axis=0)
        other_mean_color = other_edge.get_color_array(3)[other_edge_mat].mean(axis=0)
        
        return ((self_mean_color - other_mean_color) ** 2).sum()
        
    def shape_distance(self, other_edge):
        return cv2.matchShapes(self.points, other_edge.points, 2, 0.0)

    
    def color_segment_distance(self, other_edge, plot=False):
        e1 = self
        e2 = other_edge
        
        thickness = CFG['color_array_thickness']
        src = e1.get_color_array(thickness)
        dst = e2.get_color_array(thickness)
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=CFG['erosion_kernel_size'])
        
        edge_arr_mask = np.zeros(src.shape[:2]) 
        edge_arr_mask[np.any(src > 0, axis=2)] = 1
        edge_arr_mask_eroded = cv2.erode(edge_arr_mask, kernel, iterations=CFG['erosion_iterations'])
        src[edge_arr_mask_eroded == 0] = 0

        edge_arr_mask = np.zeros(dst.shape[:2]) 
        edge_arr_mask[np.any(dst > 0, axis=2)] = 1
        edge_arr_mask_eroded = cv2.erode(edge_arr_mask, kernel, iterations=CFG['erosion_iterations'])
        dst[edge_arr_mask_eroded == 0] = 0
        
        len_e1_points = len(e1.points)
        len_e2_points = len(e2.points)
        
        if CFG['use_hsv']:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
            dst = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)
        
        channel_num = src.shape[2]
        
        e1_pixels_seq = []
        e2_pixels_seq = []
        diff_seq = []
        
        e1_points = e1.points
        e2_points = e2.points[::-1]

        diff_sum = 0.0
        for i in range(len_e1_points):
            e2_i = int((len_e2_points * i) / len_e1_points)
            e1_point = e1_points[i].astype(int)
            e2_point = e2_points[e2_i].astype(int)
            
            compare_width = CFG['color_compare_pixel_width']
            e1_x_min = max(0, e1_point[1]-compare_width+1)
            e1_x_max = e1_point[1]+compare_width
            e1_y_min = max(0, e1_point[0]-compare_width+1)
            e1_y_max = e1_point[0]+compare_width
            
            e2_x_min = max(0, e2_point[1]-compare_width+1)
            e2_x_max = e2_point[1]+compare_width
            e2_y_min = max(0, e2_point[0]-compare_width+1)
            e2_y_max = e2_point[0]+compare_width
            
            
            e1_pixels = src[e1_x_min:e1_x_max, e1_y_min:e1_y_max]
            e2_pixels = dst[e2_x_min:e2_x_max, e2_y_min:e2_y_max]
            
            e1_pixels = e1_pixels.reshape(-1, channel_num)
            e1_pixels = e1_pixels[e1_pixels.sum(axis=1) > 0].mean(axis=0)
            
            e2_pixels = e2_pixels.reshape(-1, channel_num)
            e2_pixels = e2_pixels[e2_pixels.sum(axis=1) > 0].mean(axis=0)
            
            e1_pixels_seq.append(e1_pixels)
            e2_pixels_seq.append(e2_pixels)
        
        #e1_pixels_seq = cv2.blur(np.array(e1_pixels_seq)[None, :, :], (30, 30))
        #e2_pixels_seq = cv2.blur(np.array(e2_pixels_seq)[None, :, :], (30, 30))
        
        #for e1_pixels, e2_pixels in zip(e1_pixels_seq[0], e2_pixels_seq[0]):
        
        part_num = CFG['num_segments']
        segment_len = int(len(e1_pixels_seq) / part_num)
        e1_pixels_seq_avg = np.empty((part_num, channel_num))
        e2_pixels_seq_avg = np.empty((part_num, channel_num))
        for i in range(0, part_num):
            e1_pixels_seq_avg[i] = np.array(e1_pixels_seq[i*segment_len:(i+1)*segment_len]).mean(axis=0)
            e2_pixels_seq_avg[i] = np.array(e2_pixels_seq[i*segment_len:(i+1)*segment_len]).mean(axis=0)

        for e1_pixels, e2_pixels in zip(e1_pixels_seq_avg, e2_pixels_seq_avg):
            e1_pixels = e1_pixels / 255
            e2_pixels = e2_pixels / 255
            #diff = np.sqrt(((e1_pixels - e2_pixels) ** 2).sum())
            diff = ((e1_pixels - e2_pixels) ** 2).sum()
            
            diff_sum += diff
            diff_seq.append(diff)
        
        diff_sum = diff_sum / len(e2_pixels_seq_avg) #len_e1_points
            
        if plot:
            plt.figure(figsize=(10, 10))
            plt.title('src')
            plt.imshow(src, cmap='gray')
            plt.show()

            plt.figure(figsize=(10, 10))
            plt.title('dst')
            plt.imshow(dst, cmap='gray')
            plt.show()

            e1_pixels_seq = np.array(e1_pixels_seq_avg).reshape(1, -1, channel_num).astype(int)
            e1_pixels_seq = np.repeat(e1_pixels_seq, 5, axis=0)
            plt.figure(figsize=(5, 5))
            plt.imshow(e1_pixels_seq)
            plt.show()
            
            e2_pixels_seq = np.array(e2_pixels_seq_avg).reshape(1, -1, channel_num).astype(int)
            e2_pixels_seq = np.repeat(e2_pixels_seq, 5, axis=0)
            plt.figure(figsize=(5, 5))
            plt.imshow(e2_pixels_seq)
            plt.show()
            
            #print(np.concatenate([e1_pixels_seq, e2_pixels_seq, np.array(diff_seq)[None, :, None]], axis=2))
            
            plt.figure(figsize=(6, 6))
            plt.imshow(np.array(diff_seq)[None, :], cmap='gray')
            plt.show()
        
        return diff_sum
    
    def affine_distance(self, other_edge, plot=False):
        
        e1_p_first, e1_p_last = self.points[0], self.points[-1]
        e2_p_first, e2_p_last = other_edge.points[0], other_edge.points[-1]

        v1 = np.array(e1_p_first) - np.array(e1_p_last)
        v2 = e2_p_last - e2_p_first

        dot = v1[0] * v2[0] + v1[1] * v2[1]      # dot product between [x1, y1] and [x2, y2]
        det = v1[0] * v2[1] - v1[1] * v2[0]
        alpha = np.arctan2(det, dot)
        
        rotation_point = tuple(e2_p_first + v2/2.0)
        warp_mat = cv2.getRotationMatrix2D(rotation_point, 180*alpha/np.pi, 1)
        bias = e1_p_last + v1/2.0 - rotation_point
        warp_mat[:, 2] = warp_mat[:, 2] + bias
            
        aligned_e2_points = other_edge.points.dot(warp_mat[:, :2].T) + warp_mat[:, 2]
        
        zoom = self.points.shape[0] / aligned_e2_points.shape[0]
        aligned_e2_points_interp_0 = interpolation.zoom(aligned_e2_points[:, 0], zoom, mode='nearest')[:, None]
        aligned_e2_points_interp_1 = interpolation.zoom(aligned_e2_points[:, 1], zoom, mode='nearest')[:, None] 
        aligned_e2_points = np.hstack([aligned_e2_points_interp_0,aligned_e2_points_interp_1]).reshape(-1, 2)
        
        if plot:
            print('original_points', self.points[::-1])
            print('aligned points', aligned_e2_points)
            print('mean error = ', np.sqrt(((self.points[::-1] - aligned_e2_points) ** 2).sum()))
            img1 = np.zeros((300, 300, 3))

            for x,y in self.points:
                img1[int(y),int(x), :] = 255
            plt.imshow(img1)

            for x,y in aligned_e2_points:
                img1[int(y),int(x), :] = 255
            plt.imshow(img1)
            
        return np.sqrt(((self.points[::-1] - aligned_e2_points) ** 2).sum()) / self.points.shape[0] 
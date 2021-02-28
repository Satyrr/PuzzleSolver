import numpy as np
import cv2
import matplotlib.pyplot as plt

import random as rng
from itertools import combinations
from scipy.ndimage import interpolation
from .cfg import CFG
from .edge import Edge

def closest_point_idx(point, dst_points):
    x, y = point
    distances = [(x-p[0])**2 + (y-p[1])**2 for p in dst_points]
    
    return np.argmin(distances)

class Puzzle:
    
    @property
    def contours(self):
        thresh_blurred = self.thresh_blurred
        contours, hierarchy = cv2.findContours(thresh_blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        
        return contours
    
    @property
    def contours_drawing(self):
        thresh_blurred = self.thresh_blurred
        contour_drawing = np.zeros((thresh_blurred.shape[0], thresh_blurred.shape[1]), dtype=np.uint8)
        
        contours = self.contours
        for i in range(len(contours)):
            contour_color = (255,255,255)        
            cv2.drawContours(contour_drawing, contours, i, 255)
        
        return contour_drawing
    
    @property
    def puzzle_contour(self):
        """ Returns contour which is child of the root and has the biggest number of 
            points.
        """
        thresh_blurred = self.thresh_blurred
        contours, hierarchy = cv2.findContours(thresh_blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        
        contours = [c for c, h in zip(contours, hierarchy[0]) if h[3] == 0]
        puzzle_contour = contours[np.argmax([c.shape[0] for c in contours])]
        
        return puzzle_contour
    
    @property
    def hull_defects(self):
        contours = self.contours
        puzzle_contour = self.puzzle_contour
        hull = cv2.convexHull(puzzle_contour, returnPoints = False)
        defects = cv2.convexityDefects(puzzle_contour, hull)
        
        return defects
    
    @property
    def corners(self):
        thresh_blurred = self.thresh_blurred
        corners = cv2.goodFeaturesToTrack(thresh_blurred, CFG['max_corners'], 
                                  CFG['corner_quality_level'], 
                                  CFG['corner_min_distance'], 
                                  blockSize=CFG['corner_block_size'], 
                                  useHarrisDetector=True, 
                                  k=CFG['corner_k'])
        corners = np.int0(corners).squeeze()
        
        return corners
    
    @property
    def best_corners(self):
        corners = self.corners
        heads = self.heads
        
        good_corners = corners.copy()
        for h in heads:
            distances = h - good_corners[:, None, :]
            distances = (distances ** 2).sum(axis=-1)
            min_distances = np.sqrt(distances).min(axis=1)
            good_corners = good_corners[min_distances > 10]

        corner_combinations = list(combinations(good_corners.squeeze(), 4))
        areas = [cv2.contourArea(cv2.convexHull(np.array(c))) for c in corner_combinations]

        best_corners = corner_combinations[np.argmax(areas)]
        
        return best_corners
    
    @property
    def edges(self):
        if self._edges:
            return self._edges
        
        best_corners = self.best_corners
        contours = self.contours
        puzzle_contour = self.puzzle_contour
        
        puzzle_corners = list(map(tuple, best_corners))
        contour_tuples = list(map(tuple, puzzle_contour.squeeze()))

        split_indices = [closest_point_idx(puzzle_corners[i], contour_tuples) for i in range(4)]

        edges = np.split(contour_tuples, sorted(split_indices))
        if len(edges) == 5:
            edges[0] = np.vstack([edges[4], edges[0]])
            edges = edges[:-1]
            
        edge_tuples = [list(map(tuple, edges[i])) for i in range(4)]
        defects = self.hull_defects
        edge_defects = [set() for _ in range(4)]
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]

            start = tuple(puzzle_contour[s][0])
            end = tuple(puzzle_contour[e][0])
            far = tuple(puzzle_contour[f][0])
            if d > CFG['min_defect_distance']:
                for edge_idx in range(4):
                    if far in edge_tuples[edge_idx]:
                        edge_defects[edge_idx].add(far)
            
        self._edges = [Edge(self, e, d) for e, d in zip(edges, edge_defects)]
        
        return self._edges
    
    @property
    def is_corner(self):
        return sum(1 if e.edge_type == 'straight' else 0 for e in self.edges) == 2
        
    @property
    def puzzle_image_no_bg(self):
        puzzle_image = self.puzzle_image.copy()
        
        puzzle_image[self.thresh_blurred > 250] = 0
        
        return puzzle_image
    
    @property
    def heads(self):
        puzzle_contour = self.puzzle_contour
        defects = self.hull_defects
        biggest_defects = []
        headers = []
        puzzle_area = cv2.contourArea(puzzle_contour)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]

            far = tuple(puzzle_contour[f][0])
            if d > CFG['min_defect_distance']:
                biggest_defects.append((f, far))
               
        for i in range(len(biggest_defects)):
            contour_defect_idx, defect_coords = biggest_defects[i]
            
            next_defect_idx = (i+1) % len(biggest_defects)
            next_contour_defect_idx = biggest_defects[next_defect_idx][0]
            next_defect_coords = biggest_defects[next_defect_idx][1]

            puzzle_contour_list = puzzle_contour.tolist()
            
            if contour_defect_idx < next_contour_defect_idx:
                L = puzzle_contour_list[contour_defect_idx:next_contour_defect_idx]
            else:
                L = puzzle_contour_list[contour_defect_idx:] + puzzle_contour_list[:(next_contour_defect_idx+1)]
                
            convex_segment = cv2.convexHull(np.array([tuple(p[0]) for p in L]))
            convex_segment_area = cv2.contourArea(convex_segment)
            convex_segment_perimeter = cv2.arcLength(convex_segment,True)

            compactness = convex_segment_area / (convex_segment_perimeter**2)
            
            if compactness > CFG['min_header_compactness'] and convex_segment_area < puzzle_area*CFG['max_head_area']:
                headers.append(np.array(L).squeeze())
                
        return headers
    
    def __init__(self, puzzle_image, bg_threshold=230):
        self.puzzle_image = puzzle_image#cv2.cvtColor(puzzle_image, cv2.COLOR_BGR2RGB) 
        
        gray = cv2.cvtColor(self.puzzle_image, cv2.COLOR_RGB2GRAY)
        gray_blurred = cv2.medianBlur(gray, ksize=1)
        thresh = cv2.threshold(gray_blurred, bg_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh_blurred = cv2.blur(thresh, ksize=(3, 3))
        

        smoothed = cv2.blur(thresh_blurred, CFG['puzzle_smooth_kernel'])
        smoothed = cv2.inRange(smoothed, CFG['puzzle_smooth_range_min'], CFG['puzzle_smooth_range_max'])
        thresh_blurred = cv2.bitwise_not(smoothed)
        thresh_blurred[smoothed != 255] = 255
        
        
        self.gray = gray
        self.thresh = thresh
        self.thresh_blurred = thresh_blurred
        
        """ Mask threshold image with puzzle contour with some buffer"""
        mask = np.zeros(thresh_blurred.shape, np.uint8)
        puzzle_contour = self.puzzle_contour
        cv2.fillPoly(mask, pts =[puzzle_contour], color=(255))
        cv2.drawContours(mask, [puzzle_contour], 0, 255, 5)
        
        self.thresh_blurred[mask==0] = 255
        
        self._edges = []
        
    def draw_puzzle(self):
        puzzle_img = self.puzzle_image
        plt.figure(figsize=(10, 10))
        plt.imshow(puzzle_img, cmap='gray')
        plt.show()
        
    def draw_puzzle_analysis(self):
        thresh_blurred = self.thresh_blurred
        contours = self.contours
        puzzle_contour = self.puzzle_contour
        defects = self.hull_defects
        corners = self.corners
        best_corners = self.best_corners
        edges = self.edges
        
        drawing = np.zeros((thresh_blurred.shape[0], thresh_blurred.shape[1], 3), dtype=np.uint8)

        for i in range(len(contours)):
            contour_color = (255,255,255)        
            cv2.drawContours(drawing, contours, i, contour_color)
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]

            start = tuple(puzzle_contour[s][0])
            end = tuple(puzzle_contour[e][0])
            far = tuple(puzzle_contour[f][0])
            if d > CFG['min_defect_distance']:
                cv2.circle(drawing,far,2,[255,0,255],2)
                
        for i in corners:
            x,y = i.ravel()
            cv2.circle(drawing,(x,y),2,(0, 255, 255),-1)
            
        for i in best_corners:
            x,y = i.ravel()
            cv2.circle(drawing,(x,y),3,(0, 255, 0),-1)
            
        edge_colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        edge_by_type = {'straight':(255, 0, 0), 'header':(0, 0, 255), 'hole':(255, 0, 255)}
        for idx, edge in enumerate(edges):
            cv2.polylines(drawing, 
                          [np.array(edge.points)], 
                          isClosed = False,
                          color = edge_by_type[edge.edge_type], #edge_colors[idx],
                          thickness = 1)
            
        plt.figure(figsize=(10, 10))
        plt.imshow(drawing, cmap='gray')
        plt.show()
        
        
import cv2
import math
import numpy as np
from .SL import VBScanLine

class Cone:
    def __init__(self, input_data):
        if len(input_data.shape) == 2:
            self.process_image(input_data)
        else:
            raise ValueError("input_data is not a image frame.")
        
    def process_image(self, image):
        kernel_size = (5, 5)
        blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
        _, binary = cv2.threshold(blurred_image, 1, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        edges = cv2.Canny(eroded, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.top_point = None
        self.left_point = None
        self.right_point = None
        for contour in contours:
            contour_points = contour.reshape(-1, 2)
            min_y_point = contour_points[np.argmin(contour_points[:, 1])]
            min_x_point = contour_points[np.argmin(contour_points[:, 0])]
            max_x_point = contour_points[np.argmax(contour_points[:, 0])]

            if self.top_point is None or min_y_point[1] < self.top_point[1]:
                self.top_point = min_y_point
            
            if self.left_point is None or min_x_point[0] < self.left_point[0]:
                self.left_point = min_x_point
            
            if self.right_point is None or max_x_point[0] > self.right_point[0]:
                self.right_point = max_x_point

        self.top_point   = np.array(self.top_point[::-1]) if self.top_point is not None else None
        self.left_point  = np.array(self.left_point[::-1]) if self.left_point is not None else None
        self.right_point = np.array(self.right_point[::-1]) if self.right_point is not None else None
        
        if (self.left_point is not None) and (self.right_point is not None):
            self.bottom_point= (self.left_point+self.right_point)/2
        else:
            self.bottom_point=None
        self.viewport    = image.shape
    
    def left(self):
        if self.top_point is not None and self.left_point is not None:
            SL = np.vstack((self.top_point, self.left_point))
            vb  = VBScanLine(SL,image_dim=self.viewport)
            return vb.get_bound_coords()
        else:
            raise ValueError("Could not detect the Cone")

    def right(self):
        if self.top_point is not None and self.right_point is not None:
            SL = np.vstack((self.top_point, self.right_point))
            vb  = VBScanLine(SL,image_dim=self.viewport)
            return vb.get_bound_coords()
        else:
            raise ValueError("Could not detect the Cone")
    

    def mid(self):
        if self.top_point is not None and self.bottom_point is not None:
            SL = np.vstack((self.top_point, self.bottom_point))
            vb  = VBScanLine(SL,image_dim=self.viewport)
            return vb.get_bound_coords()
        else:
            raise ValueError("Could not detect the Cone")

    def angle(self,segment1, segment2):
        u = segment1[1] - segment1[0]
        v = segment2[1] - segment2[0]
        dot_product = np.dot(u, v)
        magnitude_u = np.linalg.norm(u)
        magnitude_v = np.linalg.norm(v)
        cos_theta = dot_product / (magnitude_u * magnitude_v)
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return int(angle_deg)



def get_ultrasound_cone(movies,Aidx=None,batchmode=False):
    _cones      = {'left':[],'right':[],'mid':[],'angle':[]}
    if not batchmode:
        movies = np.array([movies])
        Aidx   = np.array([Aidx])

    for movie_s,Aidx_s in zip(movies,Aidx):
        if Aidx_s is None:
            c = Cone(movie_s)
        else:
            c = Cone(movie_s[Aidx_s])
        left,right,mid = c.left(),c.right(),c.mid()
        angle      = c.angle(left,right)
        _cones['left'].append(left)
        _cones['right'].append(right)
        _cones['angle'].append(angle)
        _cones['mid'].append(mid)
    for k,v in _cones.items():
        _cones[k] = np.stack(v,axis=0) if batchmode else v[0]
    return _cones
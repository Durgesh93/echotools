import numpy as np
from scipy.interpolate import  interp1d
from sklearn.neighbors import NearestNeighbors
from .coords import Coords2D


class VBScanLine(Coords2D):
    
    def __init__(self,
                    reference,
                    S=0,
                    R=0,
                    input_fmt='coord',
                    Aidx=0,
                    raxis='middle', 
                    image_dim=(255, 255),
                    batchmode=False,
                    num_pts=None
                ):
        super().__init__(S=S,R=R,raxis=raxis,batchmode=batchmode)

        self.fmt               = input_fmt
         
        if self.batchmode:
            image_dim_s      = [image_dim[0],image_dim[1]]
            image_dim        = []
            for v in image_dim_s:
                if isinstance(v,(int,float)):
                    image_dim.append(np.full(len(reference),v))
                elif isinstance(v,(list)):
                    image_dim.append(np.array(v))
                else:
                    image_dim.append(v)
            if isinstance(Aidx,(int,float)):
                Aidx               =  np.full(len(reference),Aidx)
            elif isinstance(Aidx,(list)):
                Aidx               = np.array(Aidx)
                
        self.image_dim         = np.stack(image_dim,axis=-1)

        if num_pts is None:
            self.num_pts       = int(self.image_dim.max(axis=-1).mean())
        else:
            self.num_pts       = int(num_pts)
        
        self.Aidx              = Aidx
        
       
        if self.fmt == 'coord':
            self.refcoords         = self.transform(reference)
            self.coeffs            = self.coord2coeff(self.refcoords)
        else:
            self.refcoords         = self.transform(self.coeff2coord(reference))
            self.coeffs            = self.coord2coeff(self.refcoords)
        
       
        self.INSIDE            = 0
        self.LEFT              = 1
        self.RIGHT             = 2
        self.BOTTOM            = 4
        self.TOP               = 8

           
    def bSL(self,ds=1):
        self.nn      = []
        self.refbSL  = []
        bounds       = self.get_bound_coords()
        bounds       = np.array([bounds]) if not self.batchmode else bounds
        prob_points = []
        for bound in bounds:
            y,x = bound[:,0], bound[:,1]
            distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0) ** 2 + np.ediff1d(y, to_begin=0) ** 2))
            distance = distance / distance[-1]
            fx, fy = interp1d(distance, x), interp1d(distance, y)
            alpha = np.linspace(0, 1, self.num_pts)
            x_idx, y_idx = fx(alpha), fy(alpha)
            SL           = np.stack([y_idx, x_idx], axis=-1)
            self.nn.append(NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(SL))
            self.refbSL.append(SL)
            prob_points.append(SL)
            
        self.nn     = np.array(self.nn)
        self.refbSL = np.array(self.refbSL)
        prob_points = np.array(prob_points)
        
        ds_idx      = np.arange(0,self.num_pts,1)
        ds_idx      = [ds_idx[0]]+[*ds_idx[1:self.num_pts-1:ds]]+[ds_idx[-1]]
        prob_points = prob_points[:,ds_idx,:]
        return prob_points if self.batchmode else prob_points[0] 
    

    def mSL(self,ds=1):
        self.bSL()
        Aidx         = np.array([self.Aidx]) if not self.batchmode else self.Aidx
        Aidx         = Aidx.reshape(-1,1)
        y_coords     = self.refbSL[:,:,0]
        x_coords     = np.tile(Aidx, (1, y_coords.shape[1]))
        m_SL         = np.stack([y_coords,x_coords],axis=-1)
        ds_idx      = np.arange(0,self.num_pts,1)
        ds_idx      = [ds_idx[0]]+[*ds_idx[1:self.num_pts-1:ds]]+[ds_idx[-1]]
        m_SL         = m_SL[:,ds_idx,:]
        return m_SL if self.batchmode else m_SL[0] 

    def B2AMM_coords(self,bcoords):
        self.bSL()
        if not self.batchmode:
            bcoords = np.array([bcoords])
            Aidx    = np.array([self.Aidx])
        else:
            Aidx    = self.Aidx
        
        Aidx        = Aidx.reshape(-1,1)

        m_ycoords = []
        for idx, coord in enumerate(bcoords):
            d, i        = self.nn[idx].kneighbors(coord)
            m_ycoords.append(i.ravel())

        m_ycoords = np.array(m_ycoords)
        m_xcoords = np.tile(Aidx, (1, m_ycoords.shape[1]))
        m_coords  = np.stack([m_ycoords,m_xcoords], axis=-1).astype('float')
        return m_coords if self.batchmode else m_coords[0]

    def coord2coeff(self,coords):
        coords = np.array([coords]) if not self.batchmode else coords
        coeffs = []
        for segment,dim in zip(coords,self.image_dim):
            segment  = segment[segment[:, 0].argsort()]
            a1, a2 = segment[0], segment[1]
            if np.all(segment == 0):
                raise ValueError(f'Invalid line coordinates set data')
            if np.abs(a2[1] - a1[1]) <= np.finfo(np.float16).eps:
                coeffs.append([90, a2[1]])
            elif np.abs(a2[0] - a1[0]) <= np.finfo(np.float16).eps:
                coeffs.append([0, a2[0]])
            else:
                slope,intercept = np.polyfit(segment[:, 1], segment[:, 0],1).tolist()
                angle = np.rad2deg(np.arctan(slope))
                coeffs.append([angle,intercept])
        return np.array(coeffs) if self.batchmode else coeffs[0]
    
    def coeff2coord(self,coeffs):
        coeffs = np.array([coeffs])   if not self.batchmode else coeffs
        coords = []
        for coeff,dim in zip(coeffs,self.image_dim):
            angle,c  = coeff[0],coeff[1]
            viewport = {'x_min':0,'y_min':0,'y_max':dim[0]-1,'x_max':dim[1]-1}
            if (np.abs(angle) >= 90-np.finfo(np.float16).eps) and (np.abs(angle) <= 90+np.finfo(np.float16).eps):
                y_coords = np.array([viewport['y_min'],viewport['y_max']])
                m        = 0
                segment  = np.stack([y_coords,np.polyval([m,c],y_coords)],axis=-1)
            else:
                m        = np.tan(np.deg2rad(angle))
                x_coords = np.random.choice(np.arange(viewport['x_min'], viewport['x_max']+1), size=2, replace=False)
                segment  = np.stack([np.polyval([m,c],x_coords),x_coords],axis=-1)
            coords.append(segment)
        coords = np.array(coords)
        return coords if self.batchmode else coords[0]
    
    def B2SL_coords(self,bcoords):
        self.bSL()
        bcoords = np.array([bcoords]) if not self.batchmode else bcoords
        p_coords = []
        for idx, coord in enumerate(bcoords):
            SL          = self.refbSL[idx]
            d, i        = self.nn[idx].kneighbors(coord)
            p_coords_s  = SL[i.ravel()]
            p_coords.append(p_coords_s)
        p_coords = np.stack(p_coords, axis=0)
        return p_coords if self.batchmode else p_coords[0] 

    def _compute_code(self, x, y,viewport):
        code = self.INSIDE
        if x < viewport['x_min']: code |= self.LEFT
        elif x > viewport['x_max']: code |= self.RIGHT
        if y < viewport['y_min']: code |= self.BOTTOM
        elif y > viewport['y_max']: code |= self.TOP
        return code

    def _extend_bounds(self, w, bounds):
        a1, a2 = bounds[0], bounds[1]
        if a2[1] - a1[1] == 0:  # Vertical line
            return np.array([[a1[0]-w, a1[1]], [a2[0]+w, a2[1]]], dtype='float')
        elif a2[0] - a1[0] == 0:  # horizontal line
            return np.array([[a1[0], a1[1]+w], [a2[0], a2[1]-w]], dtype='float')
        else:  # Non-vertical
            fx     = interp1d(bounds[:, 1], bounds[:, 0], fill_value='extrapolate')
            m_diff = np.diff(bounds,axis=0).squeeze()
            m      = m_diff[0]/m_diff[1]
            shiftw = lambda c, w: np.array([fx(c[1] +w), c[1] + w])
            c1 = shiftw(a1,  -np.sign(m)*w)
            c2 = shiftw(a2,  np.sign(m)*w)
            return np.array([c1, c2])

    def _clip(self, anchor,viewport):
        y1, x1 = anchor[0]
        y2, x2 = anchor[1]
        code1, code2 = self._compute_code(x1, y1,viewport), self._compute_code(x2, y2,viewport)
        while True:
            if code1 == 0 and code2 == 0:
                return np.array([[y1, x1], [y2, x2]], dtype='float'), True
            elif (code1 & code2) != 0:
                return anchor, False
            else:
                code_out = code1 if code1 != 0 else code2
                if code_out & self.TOP:
                    x = x1 + ((x2 - x1) / (y2 - y1)) * (viewport['y_max'] - y1)
                    y = viewport['y_max']
                elif code_out & self.BOTTOM:
                    x = x1 + ((x2 - x1) / (y2 - y1)) * (viewport['y_min'] - y1)
                    y = viewport['y_min']
                elif code_out & self.RIGHT:
                    y = y1 + ((y2 - y1) / (x2 - x1)) * (viewport['x_max'] - x1)
                    x = viewport['x_max']
                elif code_out & self.LEFT:
                    y = y1 + ((y2 - y1) / (x2 - x1)) * (viewport['x_min'] - x1)
                    x = viewport['x_min']

                if code_out == code1:
                    x1, y1 = x, y
                    code1 = self._compute_code(x1, y1,viewport)
                else:
                    x2, y2 = x, y
                    code2 = self._compute_code(x2, y2,viewport)

    def get_bound_coords(self):
        if not self.batchmode:
            refcoords   = np.array([self.refcoords])
            image_dim   = np.array([self.image_dim])
        else:
            refcoords   = self.refcoords
            image_dim   = self.image_dim
        bound_coords = []
        for segment,dim in zip(refcoords,image_dim):
            segment     = segment[segment[:, 0].argsort()]
            bounds      = segment[[0, -1]]
            diag_length = np.linalg.norm(dim)
            bounds      = self._extend_bounds(10 * diag_length, bounds)
            view_bounds, valid = self._clip(bounds,viewport={'x_min':0,'y_min':0,'y_max':dim[0]-1,'x_max':dim[1]-1})
            bound_coords.append(view_bounds if valid else segment)
        bound_coords   = np.array(bound_coords)
        return bound_coords if self.batchmode else bound_coords[0] 

    def AMM2B_coords(self,amm_coords):
        self.bSL()
        amm_coords = np.array([amm_coords]) if not self.batchmode else amm_coords
        b_coords   = []
        for idx, y_pred_s in enumerate(amm_coords.astype(np.uint16)[:, :, 0]):
            SL         = self.refbSL[idx]
            b_coords_s = SL[y_pred_s]
            b_coords.append(b_coords_s)
        b_coords = np.stack(b_coords, axis=0)
        return b_coords if self.batchmode else b_coords[0]
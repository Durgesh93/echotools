import numpy as np
from scipy.interpolate import interp1d
import torch


class Coords2D:

    def __init__(self, 
                    S=0,
                    R=0,
                    raxis='middle', 
                    batchmode=False
                ):
        self.S                 = S
        self.R                 = R 
        self.raxis             = raxis
        self.batchmode         = batchmode
           
    def __shift(self,coords):
        coords = np.array([coords])   if not self.batchmode else coords
        if isinstance(self.S,(int,float)):
            self.S = np.full(coords.shape[0],self.S)

        b = coords.shape[0]
        p1, p2 = coords[:, 0, :], coords[:, 1, :]
        direction = p2 - p1
        perp_dir = np.stack([-direction[:, 1], direction[:, 0]], axis=-1)
        norm_perp_dir = perp_dir / np.linalg.norm(perp_dir, axis=1, keepdims=True)
        shift = norm_perp_dir * self.S[:, np.newaxis]
        shifted_coords = coords + shift[:, np.newaxis, :]
        return shifted_coords if self.batchmode else shifted_coords[0]


    def __rotate(self, coords):
        coords = np.array([coords])   if not self.batchmode else coords
        if isinstance(self.R,(int,float)):
            self.R = np.full(coords.shape[0],self.R)

        angles = np.radians(self.R)
        def rotation_matrix(theta):
            return np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta), np.cos(theta)]])
        
        def rotate_coords(coords, center, angles):
            translated = coords - center[:, np.newaxis, :]
            rotated = np.empty_like(translated)
            for i in range(len(angles)):
                R = rotation_matrix(angles[i])
                rotated[i] = np.dot(translated[i], R.T)
            return rotated + center[:, np.newaxis, :]
        if self.raxis == 'top':
            axis_coords = coords[:, 0, :]
        elif self.raxis == 'bottom':
            axis_coords = coords[:, -1, :]
        elif self.raxis == 'middle':
            axis_coords = np.mean(coords, axis=1) 
        else:
            raise ValueError("rotation_axis must be 'top', 'bottom', or 'middle'")
        rotated_coords = rotate_coords(coords, axis_coords, angles)
        return rotated_coords if self.batchmode else rotated_coords[0]
        
    def transform(self,coords):
        coords = self.__shift(coords)
        coords = self.__rotate(coords)
        return coords


def sargmax(norm_heatmaps,anchor_idx=None):
    norm_heatmaps     = torch.from_numpy(norm_heatmaps)
    yy, xx            = torch.meshgrid([torch.arange(norm_heatmaps.shape[-2]), torch.arange(norm_heatmaps.shape[-1])],indexing='ij')
    yy, xx            = yy.float(), xx.float()
    yy                = yy.to(norm_heatmaps.device)
    xx                = xx.to(norm_heatmaps.device)

    if anchor_idx is not None:
        anchor_idx    = torch.from_numpy(anchor_idx)
        heatmaps_y    = torch.sum(norm_heatmaps * yy, dim=-2)/torch.sum(norm_heatmaps,dim=-2)
        anchor_idx    = anchor_idx.view(heatmaps_y.shape[0],1,1).repeat(1,heatmaps_y.shape[1],1)
        yy_loc        = torch.gather(input=heatmaps_y,dim=-1,index=anchor_idx)
        scoords       = torch.cat([yy_loc,anchor_idx],dim=-1)
    else:
        yy_loc        = torch.sum(norm_heatmaps * yy, dim=[-2, -1]).view(norm_heatmaps.shape[0], norm_heatmaps.shape[1], 1)
        xx_loc        = torch.sum(norm_heatmaps * xx, dim=[-2, -1]).view(norm_heatmaps.shape[0], norm_heatmaps.shape[1], 1)
        scoords       = torch.cat([yy_loc, xx_loc], 2)
        
    return scoords.numpy()

def spreads(norm_heatmaps,coords,anchor_idx=None):
    norm_heatmaps     = torch.from_numpy(norm_heatmaps)
    coords            = torch.from_numpy(coords)

    yy, xx            = torch.meshgrid([torch.arange(norm_heatmaps.shape[-2]), torch.arange(norm_heatmaps.shape[-1])],indexing='ij')
    yy, xx            = yy.float(), xx.float()
    yy                = yy.to(norm_heatmaps.device)
    xx                = xx.to(norm_heatmaps.device)
    all_grid_coords   = torch.stack([yy, xx],dim=0).view((1,1,2)+norm_heatmaps.shape[-2:]).repeat(norm_heatmaps.shape[:2]+(1,1,1))    
    coords            = coords.view(coords.shape+(1,1))
    radius            = torch.linalg.norm(all_grid_coords - coords,dim=2)
    if anchor_idx is not None:
        anchor_idx    = torch.from_numpy(anchor_idx)
        spreads_y     = (torch.sum(norm_heatmaps * radius, dim=-2)/torch.sum(norm_heatmaps,dim=-2))
        anchor_idx    = anchor_idx.view(spreads_y.shape[0],1,1).repeat(1,spreads_y.shape[1],1)
        spreads       = torch.gather(input=spreads_y,dim=-1,index=anchor_idx).squeeze()
    else:
        spreads       = torch.sum(norm_heatmaps*radius,dim=[-2,-1])
    return spreads.numpy()
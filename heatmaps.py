import numpy as np
from scipy.stats import multivariate_normal


def generate_rotation_matrix(theta):
    """ Get a rotation matrix given an angle in radians """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def generate_heatmap(shape, mu, covariance=None):
    """
    generates a heatmap with a gaussian centered on the coordinate passed in.
    heatmap will be of size shape
    coords should be scan converted

    """
    x, y = np.mgrid[0 : shape[0] : 1, 0 : shape[1] : 1]
    xy = np.column_stack([x.flat, y.flat])
    if covariance is None:
        sigma = max(shape) / 50
        sigma = np.array([sigma, sigma])
        covariance = np.diag(sigma ** 2)
    return multivariate_normal.pdf(xy, mean=mu, cov=covariance).reshape(shape)


def generate_rotated_heatmap(mu, shape, theta=0, ratio=1, sigma=None):
    """
        Returns a heatmap for the given coordinate. Default settings return a normal gaussian heatmap.
        :param coord: the coordinate in an image of size shape x shape
        :param shape: the size of the output image
        :param theta: the angle of rotation of the heatmap
        :param ratio: the ratio between long and short axes of the heatmap
        :return: numpy array containing the heatmap
        """
    rot = generate_rotation_matrix(theta)
    small_axis = 1 / ratio
    basic = np.array([[small_axis, 0], [0, 1]])
    cov = rot.dot(basic).dot(rot.T)
    if sigma is None:
        sigma = 10 * np.max(shape)
    return generate_heatmap(shape, mu, covariance=sigma * cov)



def get_heatmaps_single(coords,masks, height,width, heatmap_ratio, heatmap_sigma,paired):
    if paired:
        segments             = coords.reshape(-1,2,2)
        masks                = masks.reshape(-1,2)
    else:
        segments             = np.stack((coords[:-1], coords[1:]), axis=1)
        masks                = np.stack((masks[:-1], masks[1:]), axis=1)
    
    angles                   = []
    for segment,mask in zip(segments,masks):
        if mask.all() == True:    
            segment_diff    =  segment[1] - segment[0]
            theta           = -np.arctan2(*segment_diff).item()
        else:
            theta           = 0
        angles.append(theta)
    

    if not paired:
        angles =  [angles[0]]+angles
    angles              = np.array(angles)
    norm_heatmaps       = []
    for segment,mask,angle in zip(segments,masks,angles):
        if mask.all() == True:
            for j, co in enumerate(segment):
                hmap = generate_rotated_heatmap(co, (height,width), angle, heatmap_ratio, heatmap_sigma)
                norm_heatmaps.append(hmap/hmap.sum())
        else:
            norm_heatmaps.append(np.zeros((height,width)))
            norm_heatmaps.append(np.zeros((height,width)))
    norm_heatmaps   = np.stack(norm_heatmaps,axis=0)
    return norm_heatmaps


def get_heatmaps(coords,mask, shape, heatmap_ratio, heatmap_sigma,paired=False,batchmode=False):
    if not batchmode:
        coords         = np.array([coords])
        height         = np.array([shape[0]])
        width          = np.array([shape[1]])
        mask           = np.array([mask])
        heatmap_ratio  = np.array([heatmap_ratio])
        heatmap_sigma  = np.array([heatmap_sigma])

    if batchmode:
        height         = shape[0]
        width          = shape[1]

        if isinstance(height,(int,float)):
            height = np.full(len(coords),height)

        if isinstance(width,(int,float)):
            width =  np.full(len(coords),width)

        if isinstance(heatmap_ratio,(float,int)):
            heatmap_ratio = np.full(len(coords),heatmap_ratio)

        if isinstance(heatmap_sigma,(float,int)):
            heatmap_sigma = np.full(len(coords),heatmap_sigma)

    norm_heatmaps = []    
    for idx in range(len(coords)):
        norm_heatmaps.append(get_heatmaps_single(coords[idx],mask[idx],height[idx], width[idx],heatmap_ratio[idx], heatmap_sigma[idx],paired))
    norm_heatmaps = np.stack(norm_heatmaps)
    return norm_heatmaps




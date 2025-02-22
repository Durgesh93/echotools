import numpy as np
from scipy.interpolate import interpn
from collections import deque

def create_test_movie(num_frames, height, width, rect_size=64):
    frames = []
    for i in range(num_frames):
        frame = 0.1 * np.ones((height, width), np.float32) * 255
        pos = int((i / (num_frames - 1)) * (min(width, height) - rect_size))
        top_left = (pos, pos)
        bottom_right = (pos + rect_size, pos + rect_size)
        if bottom_right[0] <= width and bottom_right[1] <= height:
            cv2.rectangle(frame, top_left, bottom_right, (2 * i / 3 / 3, i, i // 2), -1)
        frame = np.clip(frame, 0, 255)
        frame = frame.astype(np.uint8)
        frames.append(frame)
    return np.array(frames)
    
def clip2amm(clip,bSL,batchmode=False):
    if not batchmode:
        clip = np.array([clip])
        bSL  = np.array([bSL])
    m_image_b = []
    for clip_s,per_pts in zip(clip,bSL):
        m_image = []
        q, h, w,c = clip_s.shape
        for j in range(q):
            m_image_yx = []
            for ch in range(c):
                frame_qch = clip_s[j,:,:,ch]
                m_image_yx.append(interpn((np.arange(0, h), np.arange(0, w)), frame_qch, per_pts))
            m_image_yx = np.stack(m_image_yx,axis=-1)
            m_image.append(m_image_yx)
        
        m_image = np.stack(m_image, axis=1)
        m_image_b.append(m_image)
    m_image_b = np.array(m_image_b).astype('uint8')
    return m_image_b if batchmode else m_image_b[0]




def clip_movie(movies,Aidx,WS=64,SR=1,A=0,batchmode=False):
    if WS % 2 == 0:
        shiftl = WS//2-1
        shiftr = WS//2
    else:
        shiftl = WS // 2
        shiftr = WS // 2

    if not batchmode:
        movies = np.array([movies])
        Aidx   = np.array([Aidx])
        A      = np.array([A])

    if isinstance(A,(int,float)):
        A   = np.full(len(movies),A)

    if isinstance(Aidx,(int,float)):
        Aidx = np.full(len(movies),Aidx)

    clip_l = []
    Aidx_l = []
    for movie, anchor_idx, anchor_shift in zip(movies, Aidx, A):
        anchor_idx = int(anchor_idx)
        anchor_shift = int(anchor_shift)
        anchor_shift = np.clip(anchor_shift, -WS // 2, WS // 2)
        ids = [x.item() for x in np.arange(movie.shape[0], dtype='int')]       
        window_l = []
        window_r = []
        idsl = deque(ids.copy())
        for i in range(shiftl + anchor_shift):
            idsl.rotate(SR)
            window_l.append(idsl[anchor_idx])
        idsr = deque(ids.copy())
        for i in range(shiftr - anchor_shift):
            window_r.append(idsr[anchor_idx])
            idsr.rotate(-SR)
        window = window_l[::-1] +[anchor_idx]+ window_r
        clip = movie[window]
        clip_l.append(clip)
        Aidx_l.append(WS // 2+anchor_shift)
    movies = np.stack(clip_l, axis=0)
    Aidx   = np.stack(Aidx_l,axis=0)
    return (movies,Aidx) if batchmode else (movies[0],Aidx[0]) 
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt


def read_as_grayscale_movie(file_path):
    """
    Reads a video file (GIF, AVI, MOV, or MP4) as a grayscale movie and returns it as a numpy array.
    Each frame is converted to grayscale.
    
    Args:
        file_path (str): Path to the video file.
    
    Returns:
        np.ndarray: A numpy array of shape (num_frames, height, width) containing grayscale frames.
    """
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'gif':
        # Read GIF as grayscale
        gif = imageio.mimread(file_path, as_gray=True)
        frames = [np.array(frame) for frame in gif]
        video_array = np.array(frames)
        
    elif file_extension in ['avi', 'mov', 'mp4']:
        # Read video file using OpenCV
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {file_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_gray)
        cap.release()
        video_array = np.array(frames)
    
    else:
        raise ValueError("Unsupported file format. Please use .gif, .avi, .mov, or .mp4.")
    
    video_array = video_array.transpose(0, 1, 2)
    return video_array


def write_as_png(file_path, image):
    """
    Saves a single image as a PNG file.

    Parameters:
    - file_path (str): The full path where the PNG file will be saved, including the file name.
    - image (numpy array): The image to save, as a numpy array in grayscale or RGB format.

    Returns:
    - None
    """
    # Save the image as a PNG file
    cv2.imwrite(file_path, image)
    print(f"Image saved as PNG at: {file_path}")




def write_as_png(file_path, image=None, bSL=None, coords=None, image_dim=(255, 255)):
   
    if image is None:
        image = np.zeros(image_dim, dtype=np.uint8)

   
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    if bSL is not None:
        ax.plot(bSL[:, 1], bSL[:, 0], color='green', linewidth=1, label='Scanline')

    if coords is not None:
        ax.scatter(coords[:, 1], coords[:, 0], color='blue', s=10, label='Coordinates')

    if bSL is not None or coords is not None:
        ax.legend(loc='upper right')

    ax.axis('off')    
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Image saved with visualization at: {file_path}")



def write_as_mp4(file_path,movie,fps=1):
    out  = cv2.VideoWriter(file_path,fourcc=cv2.VideoWriter_fourcc(*'mp4v'),fps=fps, frameSize=movie.shape[1:][::-1])
    for frame in movie:
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
    out.release()
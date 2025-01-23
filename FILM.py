"""
FILM-based Video Frame Interpolation

This script uses the FILM (Frame Interpolation for Large Motion) model to generate
smooth transitions between keyframes, creating a high-quality interpolated video.
It takes a series of PNG images as input and produces an MP4 video as output.

The script performs the following main steps:
1. Loads keyframes from a specified input folder
2. Preprocesses the images
3. Applies recursive frame interpolation using the FILM model
4. Generates a video from the interpolated frames
5. Saves the output video with a unique timestamp

Usage:
Set the 'input_folder' to the directory containing your PNG keyframes
Set the 'output_folder' to the desired location for the generated video
Adjust 'fps' and 'num_recursions' parameters as needed

Note: This script requires TensorFlow, TensorFlow Hub, OpenCV, and NumPy.
"""

import os
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from glob import glob
from datetime import datetime
import time

# Load the FILM model
model = hub.load('https://tfhub.dev/google/film/1')

def preprocess_image(image_path):
    """Load and preprocess an image for the FILM model."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3) # remove alpha transparency
    img = tf.image.convert_image_dtype(img, tf.float32) 
    return img

class Interpolator:
    """Wrapper class for the FILM model to perform frame interpolation."""
    def __init__(self, align=64):
        self._model = model
        self._align = align

    def __call__(self, x0, x1, dt):
        """Interpolate between two frames at a given time step."""
        inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}
        result = self._model(inputs, training=False)
        return result['image'].numpy()

def _recursive_generator(frame1, frame2, num_recursions, interpolator):
    """Recursively generate interpolated frames between two input frames."""
    if num_recursions == 0:
        yield frame1
    else:
        time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        mid_frame = interpolator(
            np.expand_dims(frame1, axis=0), np.expand_dims(frame2, axis=0), time)[0]
        yield from _recursive_generator(frame1, mid_frame, num_recursions - 1, interpolator) # left
        yield from _recursive_generator(mid_frame, frame2, num_recursions - 1, interpolator) # right

def interpolate_recursively(frames, num_recursions, interpolator):
    """Apply recursive interpolation to a list of frames."""
    n = len(frames)
    for i in range(1, n):
        yield from _recursive_generator(frames[i - 1], frames[i], num_recursions, interpolator)
    yield frames[-1]

def process_keyframes(input_folder, output_folder, fps=30, num_recursions=3):
    """Process keyframes to create an interpolated video."""
    keyframes = sorted(glob(os.path.join(input_folder, '*.png')))
    frames = [preprocess_image(frame).numpy() for frame in keyframes]
    
    interpolator = Interpolator()
    interpolated_frames = list(interpolate_recursively(frames, num_recursions, interpolator))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = os.path.join(output_folder, f'output_video_{timestamp}.mp4')
    
    # Set up for fusing into a morphing video
    first_frame = cv2.imread(keyframes[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for frame in interpolated_frames:
        frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr) # writes
    
    out.release()
    print(f'Video created with {len(interpolated_frames)} frames: {output_video}')

# Usage
input_folder = 'vangogh_pearlgirl'
output_folder = 'FILM_Results'

start_time = time.time()
process_keyframes(input_folder, output_folder, fps=30, num_recursions=4)
end_time = time.time()

total_execution_time = end_time - start_time
print(f'Total script execution time: {total_execution_time:.2f} seconds')
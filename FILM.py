import os
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from glob import glob

# Load the FILM model
model = hub.load('https://tfhub.dev/google/film/1')

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def interpolate_frames(frame1, frame2, num_frames):
    frames = []
    for t in np.linspace(0, 1, num_frames + 2)[1:-1]:
        inputs = {
            'x0': tf.expand_dims(frame1, 0),
            'x1': tf.expand_dims(frame2, 0),
            'time': tf.constant([t], dtype=tf.float32)
        }
        output = model(inputs)
        frames.append(output['image'][0].numpy())
    return frames

def process_keyframes(input_folder, output_video, fps=30):
    keyframes = sorted(glob(os.path.join(input_folder, '*.png')))
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(keyframes[0])
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for i in range(len(keyframes) - 1):
        frame1 = preprocess_image(keyframes[i])
        frame2 = preprocess_image(keyframes[i + 1])
        
        # Interpolate between keyframes
        interpolated_frames = interpolate_frames(frame1, frame2, fps // 2)
        
        # Write frames to video
        for frame in interpolated_frames:   
            frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
    
    # Write the last keyframe
    last_frame = cv2.imread(keyframes[-1])
    out.write(last_frame)
    
    out.release()

# Usage
input_folder = 'vangogh_pearlgirl'
output_video = 'output_video.mp4'
process_keyframes(input_folder, output_video)

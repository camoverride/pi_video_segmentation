import os
import cv2
import numpy as np
import yaml


def get_most_recent_file(directory):
    """
    Get the most recent file in the directory.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    most_recent_file = max(files, key=os.path.getctime)
    return most_recent_file

def get_num_frames(memmap_file, height, width, channels=3):
    """
    Calculate the number of frames based on the size of the memmap file.
    """
    file_size = np.memmap(memmap_file, dtype='uint8', mode='r').shape[0]
    frame_size = height * width * channels
    return file_size // frame_size

def load_and_stream_memmap(memmap_file, height, width, channels=3):
    """
    Load and stream the frames from a memory-mapped file.
    """
    num_frames = get_num_frames(memmap_file, height, width, channels)
    frames = np.memmap(memmap_file,
                       dtype='uint8',
                       mode='r',
                       shape=(num_frames, height, width, channels))
    
    for i in range(num_frames):
        frame = frames[i]
        cv2.imshow('Memmap Stream', frame)

        if cv2.waitKey(300) & 0xFF == 27:  # ESC key to stop
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    composites_dir = "composites"
    most_recent_file = get_most_recent_file(composites_dir)
    print(f"Streaming the most recent file: {most_recent_file}")

    # Read the config file.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    HEIGHT = config["height"]
    WIDTH = config["width"]

    load_and_stream_memmap(memmap_file=most_recent_file, height=HEIGHT, width=WIDTH, channels=3)

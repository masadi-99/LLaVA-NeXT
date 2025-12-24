
import cv2
import base64
import os


def video_to_base64_simple(video_path):
    """
    Convert a video to a list of base64 encoded frames
    Directly turning each frame into a base64 encoded string
    """
    # Through error if the video is not found
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    return base64Frames


def video_to_base64_frames_ungridded(video_path, grid_rows=4, grid_cols=4):
    """
    Convert a video to a list of base64 encoded frames
    Breaking each frame from a 4 by 4 grid into 16 tiles and concatenate each tile in all frames into a list.
    The final result is 16 videos, each video contains one tile in all frames.
    """
    # Through error if the video is not found
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    video = cv2.VideoCapture(video_path)
    base64Frames = [[] for _ in range(16)]
    i = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        height, width = frame.shape[:2]
        tiles = []
        # Calculate tile dimensions
        tile_height = height // grid_rows
        tile_width = width // grid_cols
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculate the coordinates for this tile
                y_start = row * tile_height
                y_end = y_start + tile_height
                x_start = col * tile_width
                x_end = x_start + tile_width
                
                # Extract the tile
                tile = frame[y_start:y_end, x_start:x_end]
                tiles.append(tile)

        for t in range(len(tiles)):
            _, buffer = cv2.imencode(".jpg", tiles[t])
            base64Frames[t].append(base64.b64encode(buffer).decode("utf-8"))
        i = (i + 1) % 16
        
    video.release()
    return base64Frames


def image_to_base64(image_path):
    """
    Convert an image to a base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

import os
import json
import numpy as np
from typing import Any, List, Dict

def read_image(file_path: str) -> np.ndarray:
    """
    Read an image from a binary file.
    
    File encoding format:
    - First 4 bytes: Width (uint32)
    - Next 4 bytes: Height (uint32)
    - Remaining bytes: Raw image data (RGB, uint8)
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        np.ndarray: NumPy array of shape (height, width, 3)
        
    Raises:
        ValueError: If the file is incomplete or corrupted
        IOError: If there's an error reading the file
    """
    with open(file_path, 'rb') as file:
        width_bytes = file.read(4)
        height_bytes = file.read(4)
        
        if len(width_bytes) != 4 or len(height_bytes) != 4:
            raise ValueError("Missing image dimensions in file")
        
        width = int.from_bytes(width_bytes, byteorder='little')
        height = int.from_bytes(height_bytes, byteorder='little')
        
        expected_bytes = width * height * 3  # RGB = 3 channels
        
        image_data = file.read()
        if len(image_data) != expected_bytes:
            raise ValueError(f"Invalid image data size. Expected {expected_bytes} bytes")
        
        # Convert to NumPy array
        image = np.frombuffer(image_data, dtype=np.uint8)
        image = image.reshape((height, width, 3))
        
        return image

def read_lidar(file_path: str) -> np.ndarray:
    """
    Read LiDAR point cloud from a binary file.
    
    Each point is encoded as 16 bytes:
    - 4 bytes: X coordinate (float32)
    - 4 bytes: Y coordinate (float32)
    - 4 bytes: Z coordinate (float32)
    - 4 bytes: Intensity (float32)
    
    Args:
        file_path (str): Path to the LiDAR file
        
    Returns:
        np.ndarray: Array of shape (n_points, 4) containing points and intensities
        
    Raises:
        ValueError: If the file size is not a multiple of 16 bytes
        IOError: If there's an error reading the file
    """
    with open(file_path, 'rb') as file:
        # Read all data
        data = file.read()
        
        # Verify file size
        if len(data) % 16 != 0:
            raise ValueError("LiDAR file size must be a multiple of 16 bytes")
        
        # Convert to points array
        points = np.frombuffer(data, dtype=np.float32)
        points = points.reshape((-1, 4))  # [x, y, z, intensity]
        
        return points

def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read and parse a JSON file containing sensor data.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        List[Dict[str, Any]]: List of sensor measurements
        
    Raises:
        IOError: If there's an error reading the file
        json.JSONDecodeError: If the JSON is invalid
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data

def main():
    """
    Main function demonstrating usage of sensor data reading functions.
    """
    # Relative path construction from the script location
    # '../../data_q123' means "go up two directories from the current script location, then into data_q123"
    # This is a common pattern to reference data relative to script location
    data_folder = "../../data_q123"
    
    # Join paths using os.path.join to ensure cross-platform compatibility
    # This handles different path separators (/ on Unix, \ on Windows)
    image_path = os.path.join(data_folder, "images", "image_1701985839_633967876.bin")
    lidar_path = os.path.join(data_folder, "lidar", "lidar_1701985839_321918010.bin")
    imu_path = os.path.join(data_folder, "imu.json")
    gps_path = os.path.join(data_folder, "gps.json")

    # Read sensor data
    image = read_image(image_path)
    lidar_points = read_lidar(lidar_path)
    imu_data = read_json_file(imu_path)
    gps_data = read_json_file(gps_path)

if __name__ == "__main__":
    main()
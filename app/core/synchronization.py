import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from .read_sensor_data import read_image, read_lidar, read_json_file

def extract_timestamp_from_filename(filename: str) -> float:
    """
    Extract timestamp from filename.
    Format: 'sensor_<seconds>_<nanoseconds>.bin'
    Example: 'lidar_1701985839_321918010.bin' -> 1701985839.321918010

    Args:
        filename (str): Name of the sensor data file

    Returns:
        float: Complete timestamp in seconds with nanosecond precision

    Raises:
        ValueError: If filename format is invalid or timestamp cannot be parsed
    """
    # Extract the basename without extension
    base = os.path.splitext(filename)[0]
    parts = base.split('_')

    if len(parts) < 3:
        raise ValueError("Invalid filename format: missing seconds or nanoseconds")

    try:
        seconds = int(parts[1])
        nanoseconds = int(parts[2])

        # Convert nanoseconds to seconds (divide by 1e9)
        timestamp = seconds + (nanoseconds / 1_000_000_000)
        return timestamp

    except (ValueError, IndexError) as error:
        raise ValueError(f"Invalid timestamp format in {filename}: {error}")

def get_sensor_data_mapping(folder_path: str, read_function: Callable) -> Dict[float, Dict]:
    """
    Create a mapping between timestamps and sensor data.

    Args:
        folder_path (str): Path to the folder containing sensor files
        read_function (Callable): Function to read sensor data

    Returns:
        Dict[float, Dict]: Mapping of timestamps to sensor data and file paths
    """
    data_mapping = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith('.bin'):
            continue

        try:
            timestamp = extract_timestamp_from_filename(filename)
            # Create full path for reading the file
            file_path = os.path.join(folder_path, filename)
            sensor_data = read_function(file_path)

            # Store only the filename, not the full path
            # This makes paths more portable across different systems
            data_mapping[timestamp] = {
                'path': filename,  # Store just the filename without directory for the api response
                'data': sensor_data
            }
        except ValueError:
            # Skip files with invalid format silently
            pass

    return data_mapping

def get_sensors_time_bounds(
    image_data: Dict,
    lidar_data: Dict,
    imu_data: List[Dict],
    gps_data: List[Dict]
) -> Tuple[float, float]:
    """
    Find common time bounds across all sensors.

    Args:
        image_data: Dictionary of image data
        lidar_data: Dictionary of LiDAR data
        imu_data: List of IMU measurements
        gps_data: List of GPS measurements

    Returns:
        Tuple[float, float]: Start and end timestamps
    """
    image_times = np.array(sorted(image_data.keys()))
    lidar_times = np.array(sorted(lidar_data.keys()))
    imu_times = np.array([entry["timestamp"] for entry in imu_data])
    gps_times = np.array([entry["timestamp"] for entry in gps_data])

    start_time = max(image_times[0], lidar_times[0], imu_times[0], gps_times[0])
    end_time = min(image_times[-1], lidar_times[-1], imu_times[-1], gps_times[-1])

    return start_time, end_time

def get_valid_reference_timestamps(
    image_data: Dict,
    lidar_data: Dict,
    start_time: float,
    end_time: float
) -> Tuple[np.ndarray, str]:
    """
    Determine valid reference timestamps based on the slowest sensor.

    Args:
        image_data: Dictionary of image data
        lidar_data: Dictionary of LiDAR data
        start_time: Start of valid time range
        end_time: End of valid time range

    Returns:
        Tuple[np.ndarray, str]: Array of reference timestamps and sensor name
    """
    image_times = np.array([
        ts for ts in sorted(image_data.keys())
        if start_time <= ts <= end_time
    ])

    lidar_times = np.array([
        ts for ts in sorted(lidar_data.keys())
        if start_time <= ts <= end_time
    ])

    return (image_times, "Camera") if len(image_times) <= len(lidar_times) else (lidar_times, "LiDAR")

def find_closest_data(target_ts: float, data_dict: Dict) -> Optional[Dict]:
    """
    Find sensor data closest to target timestamp.

    Args:
        target_ts: Target timestamp
        data_dict: Dictionary of sensor data

    Returns:
        Optional[Dict]: Closest sensor data or None if dictionary is empty
    """
    if not data_dict:
        return None

    timestamps = np.array(list(data_dict.keys()))
    idx = np.searchsorted(timestamps, target_ts)

    if idx == 0:
        closest_ts = timestamps[0]
    elif idx == len(timestamps):
        closest_ts = timestamps[-1]
    else:
        before = timestamps[idx - 1]
        after = timestamps[idx]
        closest_ts = before if abs(before - target_ts) < abs(after - target_ts) else after

    return data_dict[closest_ts]

def find_closest_imu_data(target_ts: float, imu_data: List[Dict]) -> Dict:
    """
    Find IMU data closest to target timestamp using binary search.
    
    Args:
        target_ts: Target timestamp
        imu_data: List of IMU measurements (assumed to be sorted by timestamp)
        
    Returns:
        Dict: Closest IMU measurement
    
    Time Complexity: O(log n) vs O(n) for the previous version
    """
    timestamps = np.array([entry["timestamp"] for entry in imu_data])
    
    # Binary search for the insertion point (sorted list)
    idx = np.searchsorted(timestamps, target_ts)
    
    # Handle edge cases
    if idx == 0:
        return imu_data[0]
    if idx == len(timestamps):
        return imu_data[-1]
    
    # Compare distances to find closest
    if abs(timestamps[idx] - target_ts) < abs(timestamps[idx-1] - target_ts):
        return imu_data[idx]
    else:
        return imu_data[idx-1]

def interpolate_gps_data(target_ts: float, gps_data: List[Dict]) -> Dict:
    """
    Interpolate GPS data for given timestamp.
    
    Args:
        target_ts: Target timestamp
        gps_data: List of GPS measurements
        
    Returns:
        Dict: Interpolated GPS data
    """
    timestamps = np.array([entry["timestamp"] for entry in gps_data])
    
    if target_ts <= timestamps[0]:
        return gps_data[0]
    if target_ts >= timestamps[-1]:
        return gps_data[-1]
    
    idx = np.searchsorted(timestamps, target_ts)
    t1, t2 = timestamps[idx - 1], timestamps[idx]
    weight = (target_ts - t1) / (t2 - t1)
    
    entry1, entry2 = gps_data[idx - 1], gps_data[idx]
    
    return {
        "timestamp": target_ts,
        "latitude": (1 - weight) * entry1["latitude"] + weight * entry2["latitude"],
        "longitude": (1 - weight) * entry1["longitude"] + weight * entry2["longitude"],
        "altitude": (1 - weight) * entry1["altitude"] + weight * entry2["altitude"]
    }

def synchronize_all_sensors(
    reference_timestamps: np.ndarray,
    image_data: Dict,
    lidar_data: Dict,
    imu_data: List[Dict],
    gps_data: List[Dict]
) -> List[Dict]:
    """
    Synchronize data from all sensors.

    Args:
        reference_timestamps: Array of reference timestamps
        image_data: Dictionary of image data
        lidar_data: Dictionary of LiDAR data
        imu_data: List of IMU measurements
        gps_data: List of GPS measurements

    Returns:
        List[Dict]: List of synchronized sensor frames
    """
    synchronized_frames = []

    for timestamp in reference_timestamps:
        image = find_closest_data(timestamp, image_data)
        lidar = find_closest_data(timestamp, lidar_data)
        imu = find_closest_imu_data(timestamp, imu_data)
        gps = interpolate_gps_data(timestamp, gps_data)

        # Store data and filenames (not paths)
        synchronized_frames.append({
            "timestamp": timestamp,
            "image": image['data'] if image else None,
            "image_path": image['path'] if image else None,  # This is just the filename
            "lidar": lidar['data'] if lidar else None,
            "lidar_path": lidar['path'] if lidar else None,  # This is just the filename
            "imu": imu,
            "gps": gps
        })

    return synchronized_frames

def main() -> List[Dict]:
    """
    Main function executing sensor synchronization pipeline.
    
    Returns:
        List[Dict]: List of synchronized sensor frames
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up two directories to find the data folder
    # This is a common pattern to reference data relative to the script location
    data_root = os.path.join(script_dir, "..", "..", "data_q123")
    
    # Create paths for each sensor data source
    sensor_paths = {
        "lidar": os.path.join(data_root, "lidar"),
        "images": os.path.join(data_root, "images"),
        "imu": os.path.join(data_root, "imu.json"),
        "gps": os.path.join(data_root, "gps.json")
    }
    
    # Load sensor data
    image_data = get_sensor_data_mapping(sensor_paths["images"], read_image)
    lidar_data = get_sensor_data_mapping(sensor_paths["lidar"], read_lidar)
    imu_data = read_json_file(sensor_paths["imu"])
    gps_data = read_json_file(sensor_paths["gps"])
    
    # Get time bounds and reference timestamps
    start_time, end_time = get_sensors_time_bounds(image_data, lidar_data, imu_data, gps_data)
    reference_timestamps, reference_sensor = get_valid_reference_timestamps(
        image_data, lidar_data, start_time, end_time
    )
    
    # Synchronize sensors
    synchronized_frames = synchronize_all_sensors(
        reference_timestamps, image_data, lidar_data, imu_data, gps_data
    )
    
    return synchronized_frames

if __name__ == "__main__":
    synchronized_frames = main()
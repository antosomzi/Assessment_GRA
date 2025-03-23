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
    Returns a dictionary with timestamps sorted in ascending order.

    Args:
        folder_path (str): Path to the folder containing sensor files
        read_function (Callable): Function to read sensor data

    Returns:
        Dict[float, Dict]: Mapping of timestamps to sensor data and file paths, sorted by timestamp
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

    # Create a new sorted dictionary
    sorted_data_mapping = {k: data_mapping[k] for k in sorted(data_mapping.keys())}
    
    return sorted_data_mapping

def get_sensors_time_bounds(
    image_data: Dict,
    lidar_data: Dict,
    imu_data: List[Dict],
    gps_data: List[Dict]
) -> Tuple[float, float]:
    """
    Find common time bounds across all sensors.
    Assumes data is already sorted by timestamp.

    Args:
        image_data: Dictionary of image data (keys are timestamps)
        lidar_data: Dictionary of LiDAR data (keys are timestamps)
        imu_data: List of IMU measurements (sorted by timestamp)
        gps_data: List of GPS measurements (sorted by timestamp)

    Returns:
        Tuple[float, float]: Start and end timestamps
    """
    image_times = np.array(list(image_data.keys()))
    lidar_times = np.array(list(lidar_data.keys()))
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
    Assumes dictionaries have keys already sorted by timestamp.

    Args:
        image_data: Dictionary of image data (keys are timestamps)
        lidar_data: Dictionary of LiDAR data (keys are timestamps)
        start_time: Start of valid time range
        end_time: End of valid time range

    Returns:
        Tuple[np.ndarray, str]: Array of reference timestamps and sensor name
    """
    # Filter timestamps within valid range
    image_times = np.array([
        ts for ts in image_data.keys()
        if start_time <= ts <= end_time
    ])

    lidar_times = np.array([
        ts for ts in lidar_data.keys()
        if start_time <= ts <= end_time
    ])

    return (image_times, "Camera") if len(image_times) <= len(lidar_times) else (lidar_times, "LiDAR")

def find_closest_data(
    target_timestamps: np.ndarray,
    data_dict: Dict
) -> List[Dict]:
    """
    Find sensor data closest to multiple target timestamps in a vectorized way.
    Assumes dictionary keys are already sorted by timestamp.

    Args:
        target_timestamps: Array of target timestamps
        data_dict: Dictionary of sensor data with timestamps as keys

    Returns:
        List[Dict]: List of closest sensor data for each target timestamp
    """
    if not data_dict:
        return [None] * len(target_timestamps)

    data_timestamps = np.array(list(data_dict.keys()))

    indices = np.searchsorted(data_timestamps, target_timestamps)

    # Handle edge cases
    indices = np.maximum(indices, 1)
    indices = np.minimum(indices, len(data_timestamps) - 1)

    # Calculate distances to determine if before or after is closer using a vectorized approach
    before_distances = np.abs(target_timestamps - data_timestamps[indices - 1])
    after_distances = np.abs(data_timestamps[indices] - target_timestamps)

    closest_indices = np.where(before_distances <= after_distances, indices - 1, indices)
    closest_timestamps = data_timestamps[closest_indices]

    return [data_dict[ts] for ts in closest_timestamps]

def find_closest_imu_data(
    target_timestamps: np.ndarray,
    imu_data: List[Dict]
) -> List[Dict]:
    """
    Find IMU data closest to multiple target timestamps.
    Assumes imu_data is already sorted by timestamp.

    Args:
        target_timestamps: Array of target timestamps
        imu_data: List of IMU measurements (sorted by timestamp)

    Returns:
        List[Dict]: List of closest IMU measurements for each target timestamp
    """
    imu_timestamps = np.array([entry["timestamp"] for entry in imu_data])

    indices = np.searchsorted(imu_timestamps, target_timestamps)

    # Handle edge cases
    indices = np.maximum(indices, 1)
    indices = np.minimum(indices, len(imu_timestamps) - 1)
    # Calculate distances to determine if before or after is closer a vectorized approach
    before_distances = np.abs(target_timestamps - imu_timestamps[indices - 1])
    after_distances = np.abs(imu_timestamps[indices] - target_timestamps)

    closest_indices = np.where(before_distances <= after_distances, indices - 1, indices)

    return [imu_data[idx] for idx in closest_indices]

def interpolate_gps_data(
    target_timestamps: np.ndarray,
    gps_data: List[Dict]
) -> List[Dict]:
    """
    Interpolate GPS data for multiple given timestamps.
    Assumes gps_data is already sorted by timestamp.

    Args:
        target_timestamps: Array of target timestamps
        gps_data: List of GPS measurements (sorted by timestamp)

    Returns:
        List[Dict]: List of interpolated GPS data for each target timestamp
    """
    gps_timestamps = np.array([entry["timestamp"] for entry in gps_data])

    results = []

    for target_ts in target_timestamps:
        # Handle edge cases
        if target_ts <= gps_timestamps[0]:
            results.append(gps_data[0])
            continue
        if target_ts >= gps_timestamps[-1]:
            results.append(gps_data[-1])
            continue

        # Find indices for linear interpolation
        idx = np.searchsorted(gps_timestamps, target_ts)
        t1, t2 = gps_timestamps[idx - 1], gps_timestamps[idx]
        weight = (target_ts - t1) / (t2 - t1)

        entry1, entry2 = gps_data[idx - 1], gps_data[idx]

        interpolated = {
            "timestamp": target_ts,
            "latitude": (1 - weight) * entry1["latitude"] + weight * entry2["latitude"],
            "longitude": (1 - weight) * entry1["longitude"] + weight * entry2["longitude"],
            "altitude": (1 - weight) * entry1["altitude"] + weight * entry2["altitude"]
        }

        results.append(interpolated)

    return results

def synchronize_all_sensors(
    reference_timestamps: np.ndarray,
    image_data: Dict,
    lidar_data: Dict,
    imu_data: List[Dict],
    gps_data: List[Dict]
) -> List[Dict]:
    """
    Synchronize data from all sensors.
    Assumes all data structures are already sorted by timestamp.

    Args:
        reference_timestamps: Array of reference timestamps
        image_data: Dictionary of image data (keys are timestamps)
        lidar_data: Dictionary of LiDAR data (keys are timestamps)
        imu_data: List of IMU measurements (sorted by timestamp)
        gps_data: List of GPS measurements (sorted by timestamp)

    Returns:
        List[Dict]: List of synchronized sensor frames
    """
    images = find_closest_data(reference_timestamps, image_data)
    lidars = find_closest_data(reference_timestamps, lidar_data)
    imus = find_closest_imu_data(reference_timestamps, imu_data)
    gps_data_interpolated = interpolate_gps_data(reference_timestamps, gps_data)

    synchronized_frames = []
    for i, timestamp in enumerate(reference_timestamps):
        synchronized_frames.append({
            "timestamp": timestamp,
            "image": images[i]['data'] if images[i] else None,
            "image_path": images[i]['path'] if images[i] else None,
            "lidar": lidars[i]['data'] if lidars[i] else None,
            "lidar_path": lidars[i]['path'] if lidars[i] else None,
            "imu": imus[i],
            "gps": gps_data_interpolated[i]
        })
    if synchronized_frames:
        print(f"Reference timestamps: {synchronized_frames[0]['timestamp']} - {synchronized_frames[-1]['timestamp']}")
    else:
        print("Warning: No synchronized frames were generated")

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

    # Get time bounds
    start_time, end_time = get_sensors_time_bounds(image_data, lidar_data, imu_data, gps_data)
    # Get valid reference timestamps
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
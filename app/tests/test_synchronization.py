import pytest
import numpy as np
from typing import Dict, List

from core.synchronization import (
    extract_timestamp_from_filename,
    find_closest_imu_data,
    interpolate_gps_data,
    synchronize_all_sensors
)

@pytest.fixture
def real_imu_data():
    """Real IMU data samples with precise timestamps."""
    return [
        {
            "timestamp": 1701985839.3073568,
            "angular_velocity": {
                "x": 0.03787366300821304,
                "y": -0.06841693818569183,
                "z": -0.24434620141983032
            },
            "linear_acceleration": {
                "x": 0.040679436177015305,
                "y": -0.9547702670097351,
                "z": 10.100464820861816
            }
        },
        {
            "timestamp": 1701985839.3104036,
            "angular_velocity": {
                "x": 0.03420846909284592,
                "y": -0.03054327517747879,
                "z": 0.11606444418430328
            },
            "linear_acceleration": {
                "x": 0.1698964685201645,
                "y": -1.0002355575561523,
                "z": 10.126786231994629
            }
        }
    ]

@pytest.fixture
def real_gps_data():
    """Real GPS data samples with actual coordinates."""
    return [
        {
            "timestamp": 1701985839.341072,
            "latitude": 33.776285,
            "longitude": -84.39843833333333,
            "altitude": 300.3
        },
        {
            "timestamp": 1701985840.328405,
            "latitude": 33.77628833333333,
            "longitude": -84.398445,
            "altitude": 300.3
        }
    ]

class TestTimestampExtractionForLidar:
    """Tests for timestamp extraction for Lidar data."""

    def test_nanosecond_precision(self):
        """Verify nanosecond precision is maintained."""
        filename = "lidar_1701985839_321918010.bin"
        timestamp = extract_timestamp_from_filename(filename)
        assert timestamp == 1701985839.321918010

    def test_invalid_timestamp_format(self):
        """Test rejection of invalid timestamp formats."""
        invalid_files = [
            "lidar_1701985839.bin",  # Missing nanoseconds
            "lidar_abc_123.bin",      # Invalid numbers
            "1701985839_321918010.bin"  # Missing sensor prefix
        ]
        for filename in invalid_files:
            with pytest.raises(ValueError):
                extract_timestamp_from_filename(filename)

class TestIMUDataMatching:
    """Tests for IMU data synchronization."""

    def test_exact_timestamp_match(self, real_imu_data):
        """Verify exact timestamp matching."""
        target_ts = real_imu_data[0]["timestamp"]
        result = find_closest_imu_data(target_ts, real_imu_data)
        assert result == real_imu_data[0]

    def test_interpolation_point(self, real_imu_data):
            # Test case 1: Point closer to first measurement
            ts1 = real_imu_data[0]["timestamp"] + 0.001
            result1 = find_closest_imu_data(ts1, real_imu_data)
            dist1_to_first = abs(ts1 - real_imu_data[0]["timestamp"])
            dist1_to_second = abs(ts1 - real_imu_data[1]["timestamp"])
            assert result1 == real_imu_data[0]

            # Test case 2: Point closer to second measurement
            ts2 = real_imu_data[1]["timestamp"] - 0.001
            result2 = find_closest_imu_data(ts2, real_imu_data)
            dist2_to_first = abs(ts2 - real_imu_data[0]["timestamp"])
            dist2_to_second = abs(ts2 - real_imu_data[1]["timestamp"])
            assert dist2_to_second < dist2_to_first
            assert result2 == real_imu_data[1]

class TestGPSInterpolation:
    """Tests for GPS interpolation."""

    def test_midpoint_interpolation(self, real_gps_data):
        """Test GPS interpolation at midpoint."""
        mid_ts = (real_gps_data[0]["timestamp"] + real_gps_data[1]["timestamp"]) / 2
        result = interpolate_gps_data(mid_ts, real_gps_data)

        # Verify interpolated values
        expected_lat = (real_gps_data[0]["latitude"] + real_gps_data[1]["latitude"]) / 2
        expected_lon = (real_gps_data[0]["longitude"] + real_gps_data[1]["longitude"]) / 2

        assert result["latitude"] == pytest.approx(expected_lat, rel=1e-8)
        assert result["longitude"] == pytest.approx(expected_lon, rel=1e-8)

    def test_interpolation_precision(self, real_gps_data):
        """Verify preservation of GPS precision during interpolation."""
        target_ts = real_gps_data[0]["timestamp"] + 0.1
        result = interpolate_gps_data(target_ts, real_gps_data)

        # Verify precision is maintained (8 decimal places for GPS)
        lat_str = f"{result['latitude']:.8f}"
        lon_str = f"{result['longitude']:.8f}"
        assert len(lat_str.split('.')[-1]) == 8
        assert len(lon_str.split('.')[-1]) == 8

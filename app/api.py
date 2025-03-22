from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import numpy as np
from typing import List, Optional, Dict
import uvicorn
import logging

from .core.synchronization import (
    get_sensor_data_mapping,
    get_sensors_time_bounds,
    get_valid_reference_timestamps,
    synchronize_all_sensors,
    read_image,
    read_lidar,
    read_json_file
)

# Constants
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize FastAPI app
app = FastAPI(
    title="LiDAR Bike Sensor API",
    description="API for synchronizing and accessing sensor data from LiDAR bike",
    version="1.0.0"
)

#--------------------------
# Models
#--------------------------
class SensorFrame(BaseModel):
    """Model for a synchronized sensor frame"""
    timestamp: float
    image_path: Optional[str]
    lidar_path: Optional[str]
    imu: dict
    gps: dict

class SynchronizationResponse(BaseModel):
    """Response model for synchronization results"""
    total_frames: int
    reference_sensor: str
    time_range: dict
    frames: List[SensorFrame]

class SynchronizationRequest(BaseModel):
    """Request model for synchronization"""
    data_folder: str

#--------------------------
# HTML Templates
#--------------------------
def get_html_template() -> str:
    """Return the HTML template for the synchronization form"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LiDAR Bike Sensor Synchronization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .container {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
            }
            input[type="text"] {
                width: 100%;
                padding: 8px;
                margin-bottom: 20px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            .help-text {
                color: #666;
                font-size: 0.9em;
                margin-bottom: 15px;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LiDAR Bike Sensor Synchronization</h1>
            <p>Enter the path to the folder containing sensor data.</p>
            
            <form id="syncForm">
                <label for="data_folder">Data folder path:</label>
                <input type="text" id="data_folder" name="data_folder" placeholder="data_q123" required>
                <p class="help-text">
                    Enter the relative path to your data folder (e.g., "data_q123")
                </p>
                
                <button type="submit">Synchronize</button>
            </form>
            
            <div id="result" style="margin-top: 20px;"></div>
        </div>
        
        <script>
            document.getElementById('syncForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const dataFolder = document.getElementById('data_folder').value;
                const resultDiv = document.getElementById('result');
                
                resultDiv.innerHTML = '<p>Processing...</p>';
                
                try {
                    const response = await fetch('/synchronize/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ data_folder: dataFolder }),
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        resultDiv.innerHTML = `
                            <h3>Synchronization successful!</h3>
                            <p>Total frames: ${data.total_frames}</p>
                            <p>Reference sensor: ${data.reference_sensor}</p>
                            <p>Time range: ${data.time_range.start.toFixed(3)} - ${data.time_range.end.toFixed(3)}</p>
                            <p>Sample frame (first):</p>
                            <pre>${JSON.stringify(data.frames[0], null, 2)}</pre>
                        `;
                    } else {
                        resultDiv.innerHTML = `
                            <h3>Error</h3>
                            <p>${data.detail || 'An error occurred during synchronization'}</p>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <h3>Error</h3>
                        <p>An error occurred while communicating with the server.</p>
                        <p>${error.message}</p>
                    `;
                }
            });
        </script>
    </body>
    </html>
    """

#--------------------------
# Helper Functions
#--------------------------
def resolve_data_path(data_folder: str) -> str:
    """
    Resolve data folder path as relative to application directory
    
    Args:
        data_folder: Path to the data folder (relative only)
        
    Returns:
        str: Absolute path to the data folder
    """
    return os.path.join(APP_DIR, data_folder)

def verify_required_paths(data_folder: str) -> None:
    """
    Verify that all required paths exist
    
    Args:
        data_folder: Path to the data folder
        
    Raises:
        HTTPException: If any required path is missing
    """
    if not os.path.exists(data_folder):
        raise HTTPException(
            status_code=404,
            detail="Data folder not found. Please check the folder name."
        )

    lidar_folder = os.path.join(data_folder, "lidar")
    image_folder = os.path.join(data_folder, "images")
    imu_path = os.path.join(data_folder, "imu.json")
    gps_path = os.path.join(data_folder, "gps.json")

    required_paths = [
        (lidar_folder, "lidar folder"),
        (image_folder, "images folder"),
        (imu_path, "imu.json"),
        (gps_path, "gps.json")
    ]
    
    for path, name in required_paths:
        if not os.path.exists(path):
            raise HTTPException(
                status_code=404,
                detail=f"Required {name} not found in the data folder."
            )

def create_sensor_frames(
    synchronized_frames: List[Dict]
) -> List[SensorFrame]:
    """
    Create sensor frames for API response
    
    Args:
        synchronized_frames: Raw synchronized frames from synchronize_all_sensors
        
    Returns:
        List[SensorFrame]: Formatted frames for API response
    """
    formatted_frames = []
    
    for frame in synchronized_frames:
        # Create formatted frame with paths from synchronized_frames
        formatted_frame = SensorFrame(
            timestamp=frame["timestamp"],
            # Use the paths from synchronized_frames
            image_path=frame.get("image_path"),
            lidar_path=frame.get("lidar_path"),
            imu={k: v for k, v in frame["imu"].items() if k != "timestamp"},
            gps={k: v for k, v in frame["gps"].items() if k != "timestamp"}
        )
        formatted_frames.append(formatted_frame)
    
    return formatted_frames

#--------------------------
# API Routes
#--------------------------
@app.get("/synchronize/", response_class=HTMLResponse)
async def get_synchronize_form(request: Request):
    """
    Render HTML form for sensor data synchronization
    """
    return get_html_template()

@app.post("/synchronize/", response_model=SynchronizationResponse)
async def synchronize_sensor_data(request: SynchronizationRequest):
    """
    Synchronize sensor data from a given folder and return synchronized frames
    
    Args:
        request: Object containing data_folder path
        
    Returns:
        JSON response with synchronized sensor data
    """
    try:
        # Get data folder path (relative only)
        data_folder = request.data_folder
        full_data_path = resolve_data_path(data_folder)
        verify_required_paths(full_data_path)

        # Define paths
        lidar_folder = os.path.join(full_data_path, "lidar")
        image_folder = os.path.join(full_data_path, "images")
        imu_path = os.path.join(full_data_path, "imu.json")
        gps_path = os.path.join(full_data_path, "gps.json")

        # Load sensor data
        image_data = get_sensor_data_mapping(image_folder, read_image)
        lidar_data = get_sensor_data_mapping(lidar_folder, read_lidar)
        imu_data = read_json_file(imu_path)
        gps_data = read_json_file(gps_path)

        # Get time bounds
        start_time, end_time = get_sensors_time_bounds(
            image_data, lidar_data, imu_data, gps_data
        )

        # Get reference timestamps
        reference_timestamps, reference_sensor = get_valid_reference_timestamps(
            image_data, lidar_data, start_time, end_time
        )

        # Synchronize sensors
        synchronized_frames = synchronize_all_sensors(
            reference_timestamps,
            image_data,
            lidar_data,
            imu_data,
            gps_data
        )

        # Format response - using the updated function that doesn't rely on paths
        formatted_frames = create_sensor_frames(synchronized_frames)

        # Create response object
        response = SynchronizationResponse(
            total_frames=len(formatted_frames),
            reference_sensor=reference_sensor,
            time_range={
                "start": start_time,
                "end": end_time
            },
            frames=formatted_frames
        )
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Error processing sensor data. Please check your data folder and try again."
        )

#--------------------------
# Server Startup
#--------------------------
def start_server():
    """Start the API server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()
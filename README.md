# LiDAR Bike Sensor Synchronization API

This API provides synchronization capabilities for multi-sensor data collected from a bike equipped with LiDAR, camera, IMU, and GPS sensors. It processes temporal data to align measurements from different sensors with varying sampling rates.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Sensor data folder with the following structure:

```
data_q123/
├── images/        # Camera image files
├── lidar/         # LiDAR point cloud files
├── imu.json       # IMU measurements
└── gps.json       # GPS coordinates
```

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/antosomzi/GRA.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

From the project root directory, start the API server:

```bash
python -m app.api
```

The API will be available at: [http://0.0.0.0:8000/](http://0.0.0.0:8000/)

## Using the API

1. Open [http://0.0.0.0:8000/synchronize/](http://0.0.0.0:8000/synchronize/) in your browser.
2. Enter `data_q123` in the data folder input field.
3. Click "Synchronize" to process the sensor data.
4. View the results showing one sensor frame exemple.
5. You can download the full JSON by clicking the button 'Download Complete JSON.


## Running Tests

From the `/app` directory, run the tests:

```bash
python -m pytest tests/test_synchronization.py
```

## API Structure

```
app/
├── api.py                 # FastAPI application and routes
├── core/
│   ├── synchronization.py  # Core synchronization logic
│   ├── read_sensor_data.py # Sensor data parsing functions
└── tests/
    ├── test_synchronization.py # Unit tests for synchronization
```

## Architecture Notes
For simplicity, the API is currently implemented in a single file. However, to follow best practices:  

- It would be preferable to separate the API using a **Model-View-Controller (MVC)** architecture:  
  - **Models**: Define data structures and validation.  
  - **Views**: Handle HTML rendering.  
  - **Controllers**: Manage requests.  
  - **Helpers**: Store utility functions.  
  - In more complex frameworks, a dedicated **routing file** could also be introduced.
  - Here serialization is managed by fastAPI automatically

### 🛠️ Tests
- Basic **unit tests** have been included since testing is essential.  
- Possible improvements:  
  - Separate **fixtures** into a dedicated file.  
  - Implement **API tests** (more complex since it requires replicating a folder with LiDAR, images, IMU...).  
  - Add **integration tests** for the full synchronization pipeline.  
  - Consider **end-to-end tests** for web interface interactions.  

### 📦 Dependency Management
- Currently using **requirements.txt** for dependency management.  
- For more complex projects, consider using **Poetry** (`pyproject.toml` + `poetry.lock`) to better handle dependencies and separate development from production environments.  

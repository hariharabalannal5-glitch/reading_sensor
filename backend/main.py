"""
NASA C-MAPSS Engine Health Monitoring System - FastAPI Backend
CSIR - 4PI, NAL
Phase 1: Single Sensor Monitoring with Health Classification
Updated to read from train_FD001_with_RUL.csv
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from pathlib import Path

# Global Variables
df_train = None
sensor_columns = [f'sensor_{i}' for i in range(1, 22)]  # sensor_1 to sensor_21
setting_columns = ['op_setting_1', 'op_setting_2', 'op_setting_3']

def load_data():
    """Load and preprocess the training data from train_FD001_with_RUL.csv"""
    global df_train
   
    try:
        # Read the CSV file with RUL already calculated
        csv_path = 'train_FD001_with_RUL.csv'
        print(f"📂 Loading data from: {csv_path}")
       
        df_train = pd.read_csv(csv_path)
       
        # Verify required columns exist
        required_cols = ['engine_id', 'cycle'] + setting_columns + sensor_columns + ['RUL']
        missing_cols = [col for col in required_cols if col not in df_train.columns]
       
        if missing_cols:
            print(f"⚠ Warning: Missing columns: {missing_cols}")
            print(f"Available columns: {df_train.columns.tolist()}")
       
        # Calculate max_cycle for each engine (for progress bar)
        max_cycles = df_train.groupby('engine_id')['cycle'].max().to_dict()
        df_train['max_cycle'] = df_train['engine_id'].map(max_cycles)
       
        # Data summary
        print(f"✓ Data loaded successfully!")
        print(f"  - Total records: {len(df_train):,}")
        print(f"  - Unique engines: {df_train['engine_id'].nunique()}")
        print(f"  - Columns: {len(df_train.columns)}")
        print(f"  - Engine ID range: {df_train['engine_id'].min()} to {df_train['engine_id'].max()}")
        print(f"  - Sample data:")
        print(df_train.head(3))
       
    except FileNotFoundError:
        print(f"✗ Error: train_FD001_with_RUL.csv not found in backend/ directory")
        print(f"  Please ensure the file is placed in the same directory as main.py")
        df_train = None
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        df_train = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup: Load data when application starts
    print("=" * 70)
    print("NASA C-MAPSS Engine Health Monitoring System")
    print("CSIR - 4PI, NAL")
    print("=" * 70)
    load_data()
   
    yield
   
    # Shutdown
    print("\n✓ Application shutdown complete")

app = FastAPI(
    title="NASA C-MAPSS Engine Health Monitor",
    description="Turbofan Engine Predictive Maintenance System - Phase 1",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration for Frontend Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class SensorReading(BaseModel):
    engine_id: int
    cycle: int
    max_cycle: int
    setting1: float
    setting2: float
    setting3: float
    sensors: Dict[str, float]
    rul: int

class HealthStatus(BaseModel):
    status: str  # 'good', 'warning', 'danger'
    color: str
    message: str
    confidence: float

class SensorHealth(BaseModel):
    sensor_name: str
    value: float
    health: HealthStatus
    timestamp: int

def get_sensor_statistics():
    """Calculate sensor statistics for health classification"""
    if df_train is None:
        return None
   
    stats = {}
    for sensor in sensor_columns:
        if sensor in df_train.columns:
            stats[sensor] = {
                'mean': float(df_train[sensor].mean()),
                'std': float(df_train[sensor].std()),
                'min': float(df_train[sensor].min()),
                'max': float(df_train[sensor].max()),
                'q25': float(df_train[sensor].quantile(0.25)),
                'q75': float(df_train[sensor].quantile(0.75))
            }
    return stats

def classify_sensor_health(sensor_name: str, value: float, cycle: int, max_cycle: int, rul: int) -> HealthStatus:
    """
    Classify sensor health based on:
    1. RUL (Remaining Useful Life)
    2. Lifecycle degradation percentage
    3. Sensor value deviation from normal ranges
    """
   
    # Calculate degradation percentage
    degradation_percent = (cycle / max_cycle) * 100
   
    # Primary classification based on RUL and degradation
    if rul > 50 or degradation_percent < 60:
        status = "good"
        color = "green"
        message = f"Operating normally - {rul} cycles remaining"
        confidence = 0.95
    elif rul > 20 or degradation_percent < 85:
        status = "warning"
        color = "yellow"
        message = f"Monitor closely - {rul} cycles remaining"
        confidence = 0.80
    else:
        status = "danger"
        color = "red"
        message = f"Critical - Only {rul} cycles remaining"
        confidence = 0.90
   
    # Refine classification using sensor statistics
    if df_train is not None:
        sensor_stats = get_sensor_statistics()
        if sensor_name in sensor_stats:
            stats = sensor_stats[sensor_name]
           
            # Check if value is outside normal range (3 sigma rule)
            if stats['std'] > 0:
                z_score = abs(value - stats['mean']) / stats['std']
               
                if z_score > 3:  # Very unusual reading
                    if status == "good":
                        status = "warning"
                        color = "yellow"
                        message = f"Unusual sensor reading detected - {rul} cycles remaining"
                        confidence = 0.70
                    elif status == "warning":
                        status = "danger"
                        color = "red"
                        message = f"Abnormal sensor behavior - {rul} cycles remaining"
                        confidence = 0.85
   
    return HealthStatus(
        status=status,
        color=color,
        message=message,
        confidence=confidence
    )

@app.get("/")
async def root():
    """API Health Check"""
    data_status = "loaded" if df_train is not None else "not loaded"
    record_count = len(df_train) if df_train is not None else 0
   
    return {
        "message": "NASA C-MAPSS Engine Health Monitoring API",
        "status": "operational",
        "version": "1.0.0",
        "organization": "CSIR - 4PI, NAL",
        "data_status": data_status,
        "records": record_count
    }

@app.get("/api/engines")
async def get_engines():
    """Get list of available engine IDs"""
    if df_train is None:
        raise HTTPException(
            status_code=503,
            detail="Data not loaded. Please ensure train_FD001_with_RUL.csv is in the backend directory."
        )
   
    engines = sorted(df_train['engine_id'].unique().tolist())
    return {
        "engines": engines,
        "total": len(engines)
    }

@app.get("/api/engine/{engine_id}/info")
async def get_engine_info(engine_id: int):
    """Get engine information including max cycles"""
    if df_train is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
   
    engine_data = df_train[df_train['engine_id'] == engine_id]
   
    if engine_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Engine {engine_id} not found in dataset"
        )
   
    return {
        "engine_id": engine_id,
        "max_cycle": int(engine_data['max_cycle'].iloc[0]),
        "total_records": len(engine_data),
        "min_cycle": int(engine_data['cycle'].min()),
        "initial_rul": int(engine_data['RUL'].max())
    }

@app.get("/api/engine/{engine_id}/cycle/{cycle}", response_model=SensorReading)
async def get_sensor_data(engine_id: int, cycle: int):
    """Get all sensor readings for a specific engine at a specific cycle"""
   
    if df_train is None:
        raise HTTPException(
            status_code=503,
            detail="Data not loaded. Please ensure train_FD001_with_RUL.csv is in the backend directory."
        )
   
    # Get data from CSV
    data = df_train[(df_train['engine_id'] == engine_id) & (df_train['cycle'] == cycle)]
   
    if data.empty:
        # Try to find closest cycle if exact cycle doesn't exist
        engine_data = df_train[df_train['engine_id'] == engine_id]
        if engine_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Engine {engine_id} not found in dataset"
            )
       
        available_cycles = sorted(engine_data['cycle'].unique())
        raise HTTPException(
            status_code=404,
            detail=f"Cycle {cycle} not found for Engine {engine_id}. Available cycles: {min(available_cycles)} to {max(available_cycles)}"
        )
   
    row = data.iloc[0]
   
    # Build sensor dictionary (convert sensor_1 to s1 format for frontend compatibility)
    sensors = {}
    for i in range(1, 22):
        sensor_col = f'sensor_{i}'
        if sensor_col in row:
            sensors[f's{i}'] = float(row[sensor_col])
   
    return SensorReading(
        engine_id=int(row['engine_id']),
        cycle=int(row['cycle']),
        max_cycle=int(row['max_cycle']),
        setting1=float(row['op_setting_1']),
        setting2=float(row['op_setting_2']),
        setting3=float(row['op_setting_3']),
        sensors=sensors,
        rul=int(row['RUL'])
    )

@app.get("/api/sensor/{sensor_name}/health")
async def get_sensor_health(
    sensor_name: str,
    engine_id: int,
    cycle: int
) -> SensorHealth:
    """
    Get health status for a specific sensor
    Phase 1: Focus on single sensor monitoring
    """
   
    # Convert frontend format (s1) to CSV format (sensor_1)
    if sensor_name.startswith('s') and sensor_name[1:].isdigit():
        sensor_num = sensor_name[1:]
        csv_sensor_name = f'sensor_{sensor_num}'
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sensor name format. Use format like 's1', 's2', etc."
        )
   
    if csv_sensor_name not in sensor_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sensor name. Must be one of s1 to s21"
        )
   
    # Get sensor data
    sensor_data = await get_sensor_data(engine_id, cycle)
   
    # Get sensor value
    sensor_value = sensor_data.sensors[sensor_name]
   
    # Classify health
    health = classify_sensor_health(
        csv_sensor_name,
        sensor_value,
        cycle,
        sensor_data.max_cycle,
        sensor_data.rul
    )
   
    return SensorHealth(
        sensor_name=sensor_name,
        value=sensor_value,
        health=health,
        timestamp=cycle
    )

@app.get("/api/engine/{engine_id}/all-sensors/{cycle}")
async def get_all_sensors_health(engine_id: int, cycle: int):
    """Get health status for all sensors (Future: Phase 2+)"""
   
    sensor_data = await get_sensor_data(engine_id, cycle)
   
    all_health = {}
    for i in range(1, 22):
        sensor_name = f's{i}'
        csv_sensor_name = f'sensor_{i}'
       
        if sensor_name in sensor_data.sensors:
            health = classify_sensor_health(
                csv_sensor_name,
                sensor_data.sensors[sensor_name],
                cycle,
                sensor_data.max_cycle,
                sensor_data.rul
            )
            all_health[sensor_name] = {
                "value": sensor_data.sensors[sensor_name],
                "health": health.dict()
            }
   
    return {
        "engine_id": engine_id,
        "cycle": cycle,
        "rul": sensor_data.rul,
        "max_cycle": sensor_data.max_cycle,
        "sensors": all_health
    }

@app.get("/api/statistics")
async def get_statistics():
    """Get statistical information about sensors"""
    if df_train is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
   
    stats = get_sensor_statistics()
   
    # Add dataset statistics
    dataset_stats = {
        "total_records": len(df_train),
        "unique_engines": int(df_train['engine_id'].nunique()),
        "avg_cycles_per_engine": float(df_train.groupby('engine_id')['cycle'].max().mean()),
        "max_rul": int(df_train['RUL'].max()),
        "min_rul": int(df_train['RUL'].min())
    }
   
    return {
        "sensor_statistics": stats,
        "dataset_statistics": dataset_stats
    }

@app.get("/api/engine/{engine_id}/cycles")
async def get_available_cycles(engine_id: int):
    """Get list of available cycles for a specific engine"""
    if df_train is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
   
    engine_data = df_train[df_train['engine_id'] == engine_id]
   
    if engine_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Engine {engine_id} not found"
        )
   
    cycles = sorted(engine_data['cycle'].unique().tolist())
   
    return {
        "engine_id": engine_id,
        "cycles": cycles,
        "min_cycle": min(cycles),
        "max_cycle": max(cycles),
        "total_cycles": len(cycles)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
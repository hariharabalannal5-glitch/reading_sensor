"""
NASA C-MAPSS Engine Health Monitoring System - FastAPI Backend
CSIR - 4PI, NAL
Phase 1: Single Sensor Monitoring with Health Classification
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
sensor_columns = [f's{i}' for i in range(1, 22)]
setting_columns = ['setting1', 'setting2', 'setting3']

def load_data():
    """Load and preprocess the training data"""
    global df_train
    
    # Column names based on NASA C-MAPSS dataset structure
    column_names = ['engine_id', 'cycle'] + setting_columns + sensor_columns
    
    try:
        # Read the CSV file (converted from train.txt)
        # Adjust the path to where your CSV file is located
        df_train = pd.read_csv('train.csv', names=column_names)
        
        # Calculate RUL (Remaining Useful Life) for each engine
        # RUL = max_cycle - current_cycle for each engine
        max_cycles = df_train.groupby('engine_id')['cycle'].max().to_dict()
        df_train['max_cycle'] = df_train['engine_id'].map(max_cycles)
        df_train['rul'] = df_train['max_cycle'] - df_train['cycle']
        
        print(f"✓ Data loaded successfully: {len(df_train)} records")
        print(f"✓ Unique engines: {df_train['engine_id'].nunique()}")
        
    except FileNotFoundError:
        print("⚠ Warning: train.csv not found. Using simulated data.")
        df_train = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI
    Replaces deprecated @app.on_event("startup")
    """
    # Startup: Load data when application starts
    print("=" * 70)
    print("NASA C-MAPSS Engine Health Monitoring System")
    print("CSIR - 4PI, NAL")
    print("=" * 70)
    load_data()
    
    yield
    
    # Shutdown: Cleanup (if needed)
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
        stats[sensor] = {
            'mean': df_train[sensor].mean(),
            'std': df_train[sensor].std(),
            'min': df_train[sensor].min(),
            'max': df_train[sensor].max(),
            'q25': df_train[sensor].quantile(0.25),
            'q75': df_train[sensor].quantile(0.75)
        }
    return stats

def classify_sensor_health(sensor_name: str, value: float, cycle: int, max_cycle: int) -> HealthStatus:
    """
    Classify sensor health based on:
    1. Lifecycle degradation percentage
    2. Sensor value deviation from normal ranges
    """
    
    # Calculate degradation percentage
    degradation_percent = (cycle / max_cycle) * 100
    
    # Enhanced health classification logic
    if degradation_percent < 60:
        status = "good"
        color = "green"
        message = "Operating normally - All systems nominal"
        confidence = 0.95
    elif degradation_percent < 85:
        status = "warning"
        color = "yellow"
        message = "Monitor closely - Degradation detected"
        confidence = 0.80
    else:
        status = "danger"
        color = "red"
        message = "Critical - Immediate maintenance required"
        confidence = 0.90
    
    # If we have statistical data, refine classification
    if df_train is not None:
        sensor_stats = get_sensor_statistics()
        if sensor_name in sensor_stats:
            stats = sensor_stats[sensor_name]
            
            # Check if value is outside normal range (3 sigma rule)
            z_score = abs(value - stats['mean']) / stats['std']
            
            if z_score > 3:  # Very unusual reading
                if status == "good":
                    status = "warning"
                    color = "yellow"
                    message = "Unusual sensor reading detected"
                    confidence = 0.70
                elif status == "warning":
                    status = "danger"
                    color = "red"
                    message = "Abnormal sensor behavior - Critical"
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
    return {
        "message": "NASA C-MAPSS Engine Health Monitoring API",
        "status": "operational",
        "version": "1.0.0",
        "organization": "CSIR - 4PI, NAL"
    }

@app.get("/api/engines")
async def get_engines():
    """Get list of available engine IDs"""
    if df_train is None:
        # Return simulated engine list
        return {"engines": list(range(1, 101))}
    
    engines = sorted(df_train['engine_id'].unique().tolist())
    return {"engines": engines}

@app.get("/api/engine/{engine_id}/info")
async def get_engine_info(engine_id: int):
    """Get engine information including max cycles"""
    if df_train is None:
        # Return simulated data
        return {
            "engine_id": engine_id,
            "max_cycle": 192,
            "total_records": 192
        }
    
    engine_data = df_train[df_train['engine_id'] == engine_id]
    
    if engine_data.empty:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found")
    
    return {
        "engine_id": engine_id,
        "max_cycle": int(engine_data['max_cycle'].iloc[0]),
        "total_records": len(engine_data)
    }

@app.get("/api/engine/{engine_id}/cycle/{cycle}", response_model=SensorReading)
async def get_sensor_data(engine_id: int, cycle: int):
    """Get all sensor readings for a specific engine at a specific cycle"""
    
    if df_train is None:
        # Return simulated data
        max_cycle = 192
        sensors = {f's{i}': np.random.uniform(100, 2500) for i in range(1, 22)}
        
        return SensorReading(
            engine_id=engine_id,
            cycle=cycle,
            max_cycle=max_cycle,
            setting1=0.0023,
            setting2=0.0003,
            setting3=100.0,
            sensors=sensors,
            rul=max_cycle - cycle
        )
    
    # Get data from CSV
    data = df_train[(df_train['engine_id'] == engine_id) & (df_train['cycle'] == cycle)]
    
    if data.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"No data found for Engine {engine_id} at Cycle {cycle}"
        )
    
    row = data.iloc[0]
    
    # Build sensor dictionary
    sensors = {sensor: float(row[sensor]) for sensor in sensor_columns}
    
    return SensorReading(
        engine_id=int(row['engine_id']),
        cycle=int(row['cycle']),
        max_cycle=int(row['max_cycle']),
        setting1=float(row['setting1']),
        setting2=float(row['setting2']),
        setting3=float(row['setting3']),
        sensors=sensors,
        rul=int(row['rul'])
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
    
    if sensor_name not in sensor_columns:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid sensor name. Must be one of {sensor_columns}"
        )
    
    # Get sensor data
    sensor_data = await get_sensor_data(engine_id, cycle)
    
    # Get sensor value
    sensor_value = sensor_data.sensors[sensor_name]
    
    # Classify health
    health = classify_sensor_health(
        sensor_name, 
        sensor_value, 
        cycle, 
        sensor_data.max_cycle
    )
    
    return SensorHealth(
        sensor_name=sensor_name,
        value=sensor_value,
        health=health,
        timestamp=cycle
    )

@app.get("/api/engine/{engine_id}/all-sensors/{cycle}")
async def get_all_sensors_health(engine_id: int, cycle: int):
    """
    Get health status for all sensors (Future: Phase 2+)
    Currently returns single sensor for Phase 1
    """
    
    sensor_data = await get_sensor_data(engine_id, cycle)
    
    all_health = {}
    for sensor in sensor_columns:
        health = classify_sensor_health(
            sensor,
            sensor_data.sensors[sensor],
            cycle,
            sensor_data.max_cycle
        )
        all_health[sensor] = {
            "value": sensor_data.sensors[sensor],
            "health": health.dict()
        }
    
    return {
        "engine_id": engine_id,
        "cycle": cycle,
        "rul": sensor_data.rul,
        "sensors": all_health
    }

@app.get("/api/statistics")
async def get_statistics():
    """Get statistical information about sensors"""
    if df_train is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    stats = get_sensor_statistics()
    return {"sensor_statistics": stats}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",  # Use import string format for reload
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
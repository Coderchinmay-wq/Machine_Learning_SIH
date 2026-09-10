# AI-Enabled Mine Subsidence Monitoring System

## 🎯 Project Overview

This is a **machine learning prototype** for classifying mine sensor readings into three categories:
- **NORMAL**: Stable ground conditions
- **WARNING**: Developing instability detected
- **CRITICAL**: Immediate attention required

**Important**: This is a student prototype for the Smart India Hackathon 2026. It demonstrates ML pattern recognition on sensor data. **Real mine deployment would require extensive field validation, expert-labeled data, and safety engineering.**

## 🏗️ System Architecture

```
ESP32 Sensors → LoRa → Gateway → Internet → Backend (FastAPI)
                                              ↓
                                    ML Model (Random Forest)
                                              ↓
                                    Prediction → Database → Dashboard
```

## 📁 Project Structure

```
mine_ai/
│
├── data/
│   └── sensor_data.csv          # Training data (replace synthetic with real data)
│
├── models/
│   └── mine_subsidence_model.pkl  # Saved trained model
│
├── src/
│   ├── __init__.py              # Package marker
│   ├── generate_data.py         # Creates synthetic demo dataset
│   ├── preprocess.py            # Data loading & feature engineering
│   ├── train.py                 # Trains Random Forest model
│   ├── evaluate.py              # Model evaluation metrics
│   ├── predict.py               # Single prediction script
│   ├── analyzer.py              # Real-time analyzer module
│   └── api.py                   # FastAPI REST endpoint
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start Guide

### Step 1: Install Python

Download Python 3.8+ from https://www.python.org/downloads/

### Step 2: Create Virtual Environment

```bash
# Navigate to project folder
cd mine_ai

# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Generate Demo Dataset

```bash
python src/generate_data.py
```

This creates `data/sensor_data.csv` with synthetic data for testing.

### Step 5: Train the Model

```bash
python src/train.py
```

This will:
- Load the dataset
- Engineer features
- Train Random Forest
- Save model to `models/mine_subsidence_model.pkl`

### Step 6: Evaluate the Model

```bash
python src/evaluate.py
```

Shows accuracy, precision, recall, F1-score, and confusion matrix.

### Step 7: Test Single Prediction

```bash
python src/predict.py
```

Tests with sample sensor readings.

### Step 8: Start FastAPI Server

```bash
python src/api.py
```

Server runs at http://127.0.0.1:8000

### Step 9: Test API Endpoint

Use curl or Postman:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "NODE_01",
    "timestamp": "2026-09-08T10:30:00",
    "distance": 80.1,
    "tilt_x": 4.0,
    "tilt_y": 2.8,
    "vibration": 0.48,
    "temperature": 27.4,
    "humidity": 73
  }'
```

## 📊 Understanding the ML Approach

### Why Random Forest?

1. **Works well with small datasets** (common in student projects)
2. **Handles multiple sensor types** without complex preprocessing
3. **Provides probability estimates** for each class
4. **Interpretable** - can analyze which sensors matter most
5. **Fast inference** - suitable for real-time predictions

### Feature Engineering

The model doesn't just use raw sensor values. It also calculates:
- **Rate of change**: How quickly readings are changing
- **Rolling averages**: Short-term trends
- **Combined tilt**: Overall tilt magnitude

These help detect **developing instability** before it becomes critical.

### Important: Time-Series Considerations

Since this is sensor data over time:
- Training/test split is **chronological** (not random) to avoid data leakage
- Features use **past readings only** (no future information)
- Real deployment would benefit from more time-series specific models

## 🔮 Future Extensions

With more data, you could extend this to:

1. **Anomaly Detection**: Identify unusual patterns without labeled data
2. **Time-Series Forecasting**: Predict future displacement trends
3. **Sequence Models**: Use LSTM/GRU for temporal patterns
4. **Early Warning**: Detect patterns hours/days before critical events
5. **Multi-node Analysis**: Correlate readings across mine sections

## ⚠️ Safety & Scientific Disclaimer

**This prototype demonstrates ML pattern classification only.**

Real mine deployment requires:
- ✅ Substantial real mine sensor data (not synthetic)
- ✅ Expert-defined labels from geotechnical engineers
- ✅ Field validation under actual mining conditions
- ✅ Sensor calibration and reliability testing
- ✅ Integration with existing mine safety systems
- ✅ Independent safety certification
- ✅ Compliance with DGMS (Directorate General of Mines Safety) regulations

**Do NOT use this system for actual mine safety decisions without proper validation.**

## 📝 Dataset Format

Your CSV should have these columns:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Reading timestamp |
| node_id | string | Sensor node identifier |
| distance | float | Distance measurement (mm) |
| tilt_x | float | Tilt on X-axis (degrees) |
| tilt_y | float | Tilt on Y-axis (degrees) |
| vibration | float | Vibration level (g) |
| temperature | float | Temperature (Celsius) |
| humidity | float | Humidity (%) |
| status | string | Label: normal/warning/critical |

## 🎓 Learning Resources

- Scikit-learn documentation: https://scikit-learn.org/
- FastAPI documentation: https://fastapi.tiangolo.com/
- Random Forest explanation: https://scikit-learn.org/stable/modules/ensemble.html#forest

## 🏆 Smart India Hackathon 2026

**Problem Statement**: Development of an AI-enabled Low Cost Real Time Mine Subsidence Monitoring, Prediction and Early Warning System for Underground Coal Mines in India.

**Team**: [Your Team Name]
**Year**: 2026
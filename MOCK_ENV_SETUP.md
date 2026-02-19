# Mock Environment Setup Guide

## Prerequisites

Install the required packages in your conda environment:

```bash
# Create a new conda environment
conda create -n dcs-ml python=3.8
conda activate dcs-ml

# Install required packages
cd mock_env
pip install -r requirements.txt
```

## Running the Mock Environment

Navigate to the mock_env folder and run:

### 1. Interactive Mode (Manual Control)
```bash
cd mock_env
python main.py --mode interactive --steps 1000
```

**Controls:**
- `1`: Fire Active Radar Missile
- `2`: Fire IR Guided Missile
- `r`: Toggle Radar On/Off
- `s`: Toggle Radar Sweep Mode
- `a`: Toggle Autopilot
- `t`: Move Toward Enemy
- `Arrow Keys`: Move North/South/East/West
- `f`: Move Away from Enemy
- `Space`: No Operation
- `ESC`: Quit

### 2. AI Agent Mode (Rule-based)
```bash
python main.py --mode ai
```

### 3. Random Actions Mode
```bash
python main.py --mode random --steps 500
```
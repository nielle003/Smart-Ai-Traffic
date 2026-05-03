# Smart AI Traffic Light Controller
## Intelligent 4-Way Intersection Management System

---

## Project Overview

This project implements an **AI-driven adaptive traffic control system** for a 4-way intersection. Unlike traditional traffic lights that operate on fixed timers, this system uses computer vision and reinforcement learning to dynamically adjust signal timing based on real-time vehicle detection and traffic flow.

### Key Innovation
The system makes **two simultaneous decisions** at each cycle:
1. **Which direction gets the green light** (North-South axis vs East-West axis)
2. **How long the green light lasts** (5-25 seconds, scaled to traffic volume)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    4-Way Intersection                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │ Camera 1│  │ Camera 2│  │ Camera 3│  │ Camera 4│      │
│  │  (North)│  │  (South)│  │  (East) │  │  (West) │      │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │
│       │            │            │            │             │
│       └────────────┴────────────┴────────────┘             │
│                         │                                  │
│                         ▼                                  │
│              ┌─────────────────────┐                       │
│              │  YOLOv8 Object      │                       │
│              │  Detection (Vehicles)│                       │
│              └──────────┬──────────┘                       │
│                         │                                  │
│                         ▼                                  │
│              ┌─────────────────────┐                       │
│              │  Neural Network     │                       │
│              │  Decision Engine    │                       │
│              │  (Trained via RL)   │                       │
│              └──────────┬──────────┘                       │
│                         │                                  │
│                         ▼                                  │
│              ┌─────────────────────┐                       │
│              │  Arduino Hardware   │                       │
│              │  12 LEDs (4 lights) │                       │
│              └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1. **Perception Layer** (Computer Vision)
- **4 webcams** monitor each direction (North, South, East, West)
- **YOLOv8** (You Only Look Once) deep learning model detects vehicles in real-time
- Vehicle counts are extracted for each approach

### 2. **Decision Layer** (AI/Neural Network)
- **Input**: 6 values - vehicle counts (N, S, E, W) + wait times for each axis
- **Neural Network Architecture**: 6 → 32 → 32 → 22 outputs
  - 22 possible actions (11 durations × 2 directions)
  - Example: "Give North-South 15 seconds" or "Give East-West 9 seconds"
- **Training**: Reinforcement Learning with Q-learning
  - 8,000 training episodes
  - Reward function prioritizes clearing traffic and preventing long waits
  - **MAX_WAIT constraint**: No direction waits longer than 40 seconds (safety guarantee)

### 3. **Control Layer** (Hardware)
- **Arduino Uno** receives serial commands from Python controller
- **12 LEDs** represent 4 traffic lights (3 colors × 4 directions)
- **Safety sequencing**: Green → Yellow (3s) → All Red (1s) → Next Green
- **Pedestrian mode**: Activates every 3 cycles (8-second all-red phase)

---

## Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Object Detection** | YOLOv8 (Ultralytics) | Real-time vehicle detection |
| **AI Decision Engine** | Custom Neural Network (NumPy) | Adaptive signal control |
| **Training Algorithm** | Reinforcement Learning (Q-learning) | Learn optimal timing policies |
| **Computer Vision** | OpenCV | Video capture and processing |
| **Hardware Interface** | PySerial + Arduino | LED control |
| **Visualization** | Matplotlib | Training progress plots |

---

## Key Features

✅ **Adaptive Timing**: Green light duration scales with vehicle count  
✅ **Priority-Based**: Direction with more traffic gets priority  
✅ **Wait-Time Enforcement**: Maximum 40-second wait guarantee  
✅ **Pedestrian Support**: Automatic walk signals every 3 cycles  
✅ **Safety First**: Yellow transitions + all-red clearance phases  
✅ **Real-Time Operation**: Processes 4 video feeds simultaneously  

---

## Project Structure

```
Smart-Ai-Traffic/
├── train.py              # Neural network training (Reinforcement Learning)
├── controller.py         # Main real-time controller (YOLOv8 + AI decisions)
├── model_weights.npz     # Trained neural network weights
├── yolov8n.pt           # YOLOv8 pre-trained model
├── traffic_light/
│   └── traffic_light.ino # Arduino firmware (LED control)
└── requirements.txt      # Python dependencies
```

---

## How to Run (Brief)

1. **Train the AI**: `python train.py` → generates `model_weights.npz`
2. **Upload Arduino code**: Flash `traffic_light.ino` to Arduino Uno
3. **Run the controller**: `python controller.py` (requires 4 webcams)

---

## Academic Context

This project demonstrates the application of:
- **Deep Learning** for real-time object detection
- **Reinforcement Learning** for control optimization
- **Computer Vision** for traffic monitoring
- **Embedded Systems** integration (Arduino)
- **Intelligent Transportation Systems (ITS)** concepts

### Potential Applications
- Smart cities and adaptive traffic management
- Reducing congestion and idle time at intersections
- Emergency vehicle priority (can be extended)
- Integration with larger smart city infrastructure

---

*Note: This is a research/educational implementation demonstrating AI-driven traffic control concepts.*

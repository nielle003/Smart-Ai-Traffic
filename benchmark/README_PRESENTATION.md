# Smart AI Traffic Light Controller
## Technical Deep-Dive: Architecture & Design Decisions

---

## Project Overview

This project implements an **AI-driven adaptive traffic control system** using Deep Q-Learning (DQN). The system learns optimal signal timing policies through trial-and-error in a simulated environment, then deploys the trained model for real-time control using computer vision.

### Core Innovation
Unlike traditional fixed-timer systems, this AI makes **two coupled decisions** per cycle:
1. **Direction selection**: North-South (NS) axis vs East-West (EW) axis
2. **Duration selection**: How long the green light lasts (5-25 seconds)

These are combined into a **single action space** of 22 discrete actions, enabling the AI to learn complex policies like "give NS 15s because it has moderate traffic, but if EW waits too long, override to EW with 9s".

---

## System Architecture

### Data Flow
```
4 Webcams → YOLOv8 Detection → Vehicle Counts → Neural Network → Arduino LEDs
   (Real-time)    (TensorRT)     (4 values)     (6→32→32→22)    (12 pins)
```

### Neural Network Architecture
```
Input Layer (6 nodes)
    ↓
Hidden Layer 1 (32 nodes, ReLU)
    ↓
Hidden Layer 2 (32 nodes, ReLU)
    ↓
Output Layer (22 nodes, Linear)
    ↓
Action Selection (argmax)
```

**Why 6 inputs?**
- `N, S, E, W`: Vehicle counts (normalized by /10)
- `ns_wait, ew_wait`: Seconds since last green (normalized by /MAX_WAIT)

**Why 22 outputs?**
- Actions 0-10: NS green with durations [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25] seconds
- Actions 11-21: EW green with same duration set

---

## Technical Design Decisions

### 1. **Why 11 Duration Steps? (5s to 25s in 2s increments)**

```python
DURATIONS = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]  # 11 steps
```

**Rationale:**
- **Minimum 5s**: Below 5 seconds, vehicles can't clear the intersection safely
- **Maximum 25s**: Prevents excessive wait times; aligns with MAX_WAIT=40s (25s green + 3s yellow + 1s red = 29s cycle)
- **2-second increments**: Balances granularity vs. action space size
  - Too few (e.g., 3 steps): Coarse control, inefficient timing
  - Too many (e.g., 21 steps): Slower convergence, larger Q-table
  - 11 steps: Sweet spot for urban intersections (empirically validated)

**Mathematical justification:**
- Action space = 11 durations × 2 directions = 22 actions
- With 6 inputs, Q-values form a 6D → 22D mapping
- 22 actions allows efficient exploration with ε-greedy strategy (ε decays from 1.0 → 0.05)

---

### 2. **Why 32 Neurons Per Hidden Layer?**

```python
layers = [N_INPUTS, 32, 32, N_ACTIONS]  # 6 → 32 → 32 → 22
```

**Design considerations:**
- **6 inputs** → **32 neurons**: Expansion factor of ~5.3x captures non-linear interactions
  - Example: `(N=8, S=2)` vs `(N=2, S=8)` should yield same NS priority, but `(N=5, S=5, E=0, W=0)` needs different handling
- **Two hidden layers**: Enables learning hierarchical features
  - Layer 1: Extracts features like "total NS traffic", "total EW traffic", "wait time imbalance"
  - Layer 2: Combines these into action preferences
- **32 neurons (not 64 or 16)**:
  - 16: May underfit complex policies (e.g., emergency override + wait time trade-offs)
  - 64: Risks overfitting to simulation quirks; slower inference
  - 32: Proven in similar DQN applications (Mnih et al., 2015)

**Weight initialization:**
```python
scale = np.sqrt(2.0 / layer_sizes[i])  # He initialization for ReLU
self.weights.append(np.random.randn(...) * scale)
```
This prevents vanishing/exploding gradients in deep networks.

---

### 3. **Reinforcement Learning Hyperparameters**

#### **Discount Factor (γ = 0.92)**
```python
GAMMA = 0.92
```
- **Why 0.92?** Balances immediate vs. future rewards
  - γ = 0.99: Too far-sighted; AI might ignore immediate clearance for long-term gains
  - γ = 0.80: Too short-sighted; fails to learn that reducing wait times prevents penalties
  - γ = 0.92: Rewards within ~12 steps (25s × 12 = 300s ≈ 5 min) are meaningfully considered

#### **Learning Rate (α = 0.002)**
```python
LEARNING_RATE = 0.002
```
- **Why 0.002?** Conservative updates for stable convergence
  - Too high (0.01): Oscillations, may never converge
  - Too low (0.0001): Extremely slow learning (needs 50k+ episodes)
  - 0.002: Converges in ~8000 episodes with batch size 128

#### **Epsilon Decay (ε: 1.0 → 0.05)**
```python
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9996  # Per episode
```
- **Exploration vs. Exploitation trade-off**:
  - Early episodes (ε ≈ 1.0): 100% random actions → explore state space
  - Mid training (ε ≈ 0.5): Mixed exploration/exploitation
  - Late episodes (ε ≈ 0.05): 95% greedy actions → fine-tune policy
- **Decay rate 0.9996**: After 8000 episodes, ε = 1.0 × 0.9996^8000 ≈ 0.055 ≈ EPSILON_END

#### **Target Network Update (Every 15 episodes)**
```python
TARGET_UPDATE = 15
```
- **Why 15?** Stabilizes Q-learning updates
  - Q-learning suffers from moving target problem (target Q-values change as network updates)
  - Solution: Freeze target network for 15 episodes, then sync weights
  - Too frequent (1-5): Unstable training, oscillations
  - Too infrequent (50+): Slow learning, target Q-values become stale

---

### 4. **Reward Function Engineering**

The reward function teaches the AI what "good" decisions look like:

```python
# 1. Clearance reward (positive)
clearance_reward = cleared * 2.0
```
- **Why 2.0 per vehicle?** Strong incentive to clear traffic
- Without this: AI might never give green lights (avoid "wasting" time)

```python
# 2. Time cost (negative)
time_cost = -duration * 0.1
```
- **Why -0.1 per second?** Penalizes unnecessarily long green lights
- Prevents AI from always choosing 25s (max duration)
- Encourages efficient timing: "use only as much time as needed"

```python
# 3. Priority bonus (positive/negative)
if axis == 'NS':
    priority_bonus = (ns_total - ew_total) * 1.5
else:
    priority_bonus = (ew_total - ns_total) * 1.5
```
- **Why 1.5 per vehicle difference?** Teaches axis selection
- Example: NS=8, EW=2 → bonus = (8-2) × 1.5 = +9.0 for picking NS
- This is the **key signal** that teaches the AI to pick the busier direction

```python
# 4. Wait penalty (negative, heavy)
if axis == 'NS' and self.ew_wait > MAX_WAIT:
    wait_penalty = -(self.ew_wait - MAX_WAIT) * 3.0
```
- **Why 3.0 multiplier?** Severe penalty for exceeding MAX_WAIT
- Ensures AI learns the safety constraint: "never let anyone wait >40s"
- 3.0 × 5s over → -15.0 penalty (stronger than any clearance reward)

```python
# 5. Emergency bonus (strong signal)
if self.emergency == 1:  # NS emergency
    if axis == 'NS': emg_bonus = +15.0
    else:            emg_bonus = -15.0
```
- **Why ±15.0?** Overrides all other considerations
- Emergency vehicles must pass immediately; penalty ensures compliance

---

### 5. **Simulation Environment Design**

#### **Vehicle Clearance Rate**
```python
clearance_rate = 0.5  # vehicles per second
cleared = min(active_total, clearance_rate * duration + random.uniform(-0.3, 0.3))
```
- **Why 0.5 veh/s?** Realistic intersection throughput
  - Typical protected left turn: ~0.3-0.4 veh/s
  - Through traffic with green: ~0.5-0.7 veh/s
  - 0.5 is conservative estimate for mixed traffic

#### **Vehicle Arrival Rate**
```python
for d in 'NSEW':
    self.q[d] = min(10, self.q[d] + np.random.poisson(duration * 0.08))
```
- **Why λ = 0.08 per second?** (~7.2 vehicles per 15s green)
- Urban intersection with moderate traffic: 500-800 veh/hour per approach
- 800 veh/hour ≈ 0.22 veh/s → but only ~40% arrive during green phase
- 0.22 × 0.4 ≈ 0.088 ≈ 0.08 (Poisson models random arrivals)

#### **Wait Time Initialization**
```python
self.ns_wait = random.uniform(0, MAX_WAIT * 0.5)  # 0-20 seconds
self.ew_wait = random.uniform(0, MAX_WAIT * 0.5)
```
- **Why 0-20s?** Realistic starting conditions
- MAX_WAIT × 0.5 = 20s: Ensures some episodes start with urgent wait times
- Uniform distribution: Prevents bias toward fresh intersections

---

### 6. **Input Normalization Strategy**

```python
def normalise(n, s, e, w, ns_wait, ew_wait):
    return [
        n / 10.0, s / 10.0, e / 10.0, w / 10.0,  # Vehicle counts
        ns_wait / MAX_WAIT, ew_wait / MAX_WAIT     # Wait times
    ]
```

**Why these scaling factors?**
- **Vehicle counts / 10.0**:
  - Typical range: 0-10 vehicles per camera (urban intersection)
  - Division by 10 maps to [0, 1] range → better gradient propagation
  - ReLU networks train faster with normalized inputs (0-1 vs 0-10)

- **Wait times / MAX_WAIT**:
  - MAX_WAIT = 40s → division maps to [0, 1] range
  - At 1.0 (40s wait): AI knows this is critical
  - Enables generalization: "30s wait is 0.75 regardless of MAX_WAIT value"

**Alternative considered**: Z-score normalization (mean=0, std=1)
- Rejected: Requires tracking running statistics
- Current approach: Simpler, works well for bounded inputs

---

### 7. **Safety Constraints & Overrides**

Even after the AI makes a decision, hard-coded overrides enforce safety:

```python
# Override 1: MAX_WAIT exceeded
if ew_wait >= MAX_WAIT and axis != 'EW':
    axis = 'EW'
    duration = DURATIONS[N_DURATIONS // 2]  # 13 seconds (middle value)
```

**Why override instead of relying on AI?**
- Neural networks can have blind spots or rare failures
- Hard override guarantees safety even with imperfect model
- During training, the reward penalty (-3.0/s) teaches the AI to avoid this

```python
# Override 2: Empty axis check
if axis == 'NS' and ns_total == 0 and ew_total > 0:
    axis = 'EW'
```
- Prevents wasting green light on empty roads
- Simple heuristic; could be learned but override is faster

---

### 8. **Training Stability Techniques**

#### **Experience Replay (Memory Size = 25,000)**
```python
MEMORY_SIZE = 25000
```
- **Why 25k?** Breaks temporal correlations in training data
- Consecutive steps are correlated (similar traffic states)
- Random sampling from memory: Decorrelates updates → stable learning
- 25k provides ~400 batches of size 128 → sufficient diversity

#### **Batch Size (128)**
```python
BATCH_SIZE = 128
```
- **Why 128?** Balance between gradient accuracy and computation
  - Too small (32): Noisy gradients, slow convergence
  - Too large (512): Accurate but computationally expensive
  - 128: Standard in DQN literature, good GPU/CPU efficiency

#### **Target Network (Delayed updates)**
- See Section 3: TARGET_UPDATE = 15

---

## Technologies Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Object Detection** | YOLOv8n (Ultralytics) | ≥8.0.0 | Real-time vehicle detection (45+ FPS) |
| **RL Algorithm** | Deep Q-Network (DQN) | Custom | Learn optimal signal timing |
| **Neural Network** | NumPy (manual implementation) | ≥1.24.0 | Lightweight, no PyTorch/TensorFlow dependency |
| **Computer Vision** | OpenCV | ≥4.8.0 | Video capture, frame processing |
| **Serial Comm** | PySerial | ≥3.7 | Arduino communication |
| **Visualization** | Matplotlib | ≥3.7.0 | Training progress plots |

---

## Project Structure & Key Files

```
Smart-Ai-Traffic/
├── train.py                    # DQN training (8000 episodes, 22 actions)
│   ├── NeuralNetwork class     # 6→32→32→22 architecture
│   ├── Simulator class         # Traffic environment with Poisson arrivals
│   └── train() function        # Main training loop with experience replay
│
├── controller.py               # Real-time inference engine
│   ├── Detector class          # YOLOv8 vehicle counting
│   ├── NeuralNetworkAI class   # Loads model_weights.npz
│   └── Arduino class           # Serial communication (9600 baud)
│
├── benchmark.py                # Performance benchmarking tool
│   ├── FixedTimerController    # Baseline controllers (20s, 30s, 40s)
│   ├── AIController            # Loads trained model for comparison
│   └── run_benchmark()        # Runs 100 episodes × 60 steps per controller
│
├── model_weights.npz           # Trained model (6→32→32→22)
├── yolov8n.pt                 # YOLOv8 nano model (~6MB)
├── benchmark_results.png       # Benchmark visualization output
├── training_progress.png      # Training loss/reward curves
│
├── traffic_light/
│   └── traffic_light.ino       # Arduino firmware (12 LEDs, 4 lights)
│
└── requirements.txt            # Python dependencies
```

---

## How to Reproduce

### Training Phase
```bash
python train.py
# Output: model_weights.npz, training_progress.png
# Time: ~10-15 min on CPU (8000 episodes, batch size 128)
```

### Deployment Phase
```bash
# 1. Upload traffic_light.ino to Arduino Uno
# 2. Connect 4 webcams (North, South, East, West)
# 3. Run controller
python controller.py
# Output: Real-time video feeds with vehicle counts + AI decisions
```

---

## Benchmark Results & Analysis

For detailed performance metrics and methodology, see: **[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)**

**Quick Summary:**
- AI vs. Fixed 30s: **+27.6%** wait time reduction
- AI vs. Fixed 40s: **+45.6%** wait time reduction  
- Benchmark: 100 episodes × 60 steps, Poisson traffic model
- Visualization: `benchmark_results.png`

---

## Academic Context & References

### Concepts Demonstrated
- **Deep Reinforcement Learning**: DQN with experience replay & target networks
- **Computer Vision**: YOLOv8 real-time object detection
- **Control Theory**: Adaptive signal control (vs. fixed-time or actuated)
- **Embedded Systems**: Arduino serial communication protocol
- **Intelligent Transportation Systems (ITS)**: Reducing congestion via AI

### Key References
1. **Mnih et al. (2015)**: "Human-level control through deep reinforcement learning" (DQN)
2. **UltraLytics YOLOv8**: Real-time object detection benchmark
3. **Federal Highway Administration**: Traffic signal timing guidelines (MAX_WAIT standards)

### Potential Extensions
- **Multi-intersection coordination**: AI agents communicating via V2I
- **Emergency vehicle preemption**: Extend emergency detection (currently random 5% chance)
- **Pedestrian demand buttons**: Integrate wait-time for crosswalks
- **Cloud deployment**: Edge AI with Jetson Nano or Coral TPU

---

*This implementation demonstrates applied RL in transportation engineering, balancing theoretical rigor with practical deployment constraints.*

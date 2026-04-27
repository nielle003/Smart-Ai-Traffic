# AI Traffic Light Controller — Setup Guide

## Python dependencies
```
pip install opencv-python pyserial numpy
```

## Arduino wiring (per traffic light)

### Light A (Side A — e.g. North)
| Arduino Pin | Wire to         |
|-------------|-----------------|
| Pin 2       | RED LED + 220Ω  |
| Pin 3       | YELLOW LED + 220Ω |
| Pin 4       | GREEN LED + 220Ω |
| GND         | All LED grounds |

### Light B (Side B — e.g. East)
| Arduino Pin | Wire to         |
|-------------|-----------------|
| Pin 5       | RED LED + 220Ω  |
| Pin 6       | YELLOW LED + 220Ω |
| Pin 7       | GREEN LED + 220Ω |
| GND         | All LED grounds |

> Always use a 220Ω resistor in series with each LED to protect the Arduino pins.

## Steps to run

1. Open `traffic_light.ino` in Arduino IDE
2. Upload to Arduino Uno
3. Plug in both webcams
4. Run:  python controller.py
5. Press Q in the display window to quit safely (sends BOTH:RED before exit)

## Tuning tips

In controller.py, adjust these constants at the top:

| Constant       | What it does                                      |
|----------------|---------------------------------------------------|
| MIN_GREEN      | Minimum green time in seconds (default 5)         |
| MAX_GREEN      | Maximum green time in seconds (default 20)        |
| YELLOW_TIME    | How long yellow shows before switching (default 2)|
| MOG_THRESHOLD  | Lower = more sensitive to motion (default 50)     |
| MIN_BLOB_AREA  | Smaller = counts smaller objects (default 800)    |

## How the AI works (for your presentation)

1. **Perception**: Background subtraction (MOG2) isolates moving blobs per camera.
   Blobs above MIN_BLOB_AREA are counted as vehicles.
   A rolling 5-frame average smooths out jitter.

2. **Decision**: Whichever side has more vehicles gets green.
   Green duration scales with count:
     duration = MIN_GREEN + (count / 10) × (MAX_GREEN - MIN_GREEN)
   Clamped between MIN_GREEN and MAX_GREEN.

3. **Safety**: Yellow always runs before a phase switch.
   Both lights go RED for 1 second between every phase change.
   On quit/crash, Arduino receives BOTH:RED.

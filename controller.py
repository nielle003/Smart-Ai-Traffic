"""
4-Way AI Traffic Controller — 4 Camera Edition
================================================
One camera per direction: North(0), South(1), East(2), West(3).
Neural network sees all 4 directions independently.

Signal cycle:
  NS GREEN (NN decision) → NS YELLOW (3s) → ALL RED (1s)
  EW GREEN (NN decision) → EW YELLOW (3s) → ALL RED (1s)
  PEDESTRIAN every 3 cycles (8s) → ALL RED (1s) → repeat

Requirements:
  pip install opencv-python pyserial numpy ultralytics

Run order:
  1. python train.py          → model_weights.npz
  2. Upload traffic_light.ino → Arduino
  3. python controller.py
"""

import cv2
import serial
import serial.tools.list_ports
import numpy as np
import time
import sys
import threading
import os

# ── Config ────────────────────────────────────────────────────────────────────
# 4 cameras: one per direction
CAMERA_N = 1
CAMERA_S = 2
CAMERA_E = 3
CAMERA_W = 4

SERIAL_PORT        = None
BAUD_RATE          = 9600

YELLOW_TIME        = 3
ALL_RED_TIME       = 1
PEDESTRIAN_TIME    = 8
PED_EVERY_N_CYCLES = 3

VEHICLE_CLASSES    = [2, 3, 5, 7]

MIN_DURATION   = 5.0
MAX_DURATION   = 25.0
DURATION_STEPS = np.linspace(MIN_DURATION, MAX_DURATION, 21)
N_ACTIONS      = len(DURATION_STEPS)
N_INPUTS       = 7  # added phase

# ── Neural Network ────────────────────────────────────────────────────────────

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes)-1):
            scale = np.sqrt(2.0/layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1])*scale)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def relu(self, x): return np.maximum(0, x)

    def predict(self, x):
        cur = np.array(x, dtype=float).reshape(1,-1)
        for i,(w,b) in enumerate(zip(self.weights, self.biases)):
            z   = cur @ w + b
            cur = z if i==len(self.weights)-1 else self.relu(z)
        return cur

    def load(self, path):
        data = np.load(path)
        self.weights = [data[f'w{i}'] for i in range(len(self.weights))]
        self.biases  = [data[f'b{i}'] for i in range(len(self.biases))]

def normalise(n, s, e, w, wait_time, emergency, phase):
    return [n/10.0, s/10.0, e/10.0, w/10.0, wait_time/40.0, emergency/2.0, phase]

# ── Arduino ───────────────────────────────────────────────────────────────────

def find_arduino():
    for p in serial.tools.list_ports.comports():
        if any(k in p.description for k in ["Arduino","CH340","USB Serial"]):
            return p.device
    ports = serial.tools.list_ports.comports()
    return ports[0].device if ports else None

class Arduino:
    def __init__(self, port):
        print(f"[Arduino] Connecting on {port}...")
        self.ser   = serial.Serial(port, BAUD_RATE, timeout=2)
        self._lock = threading.Lock()
        time.sleep(2)
        print(f"[Arduino] {self.ser.readline().decode().strip()}")
        self.both_red()

    def _send(self, cmd):
        with self._lock:
            self.ser.write((cmd+"\n").encode())
            r = self.ser.readline().decode().strip()
            if r != "OK":
                print(f"[Arduino] Warning: '{r}' for '{cmd}'")

    def set_ns(self, state): self._send(f"NS:{state}")
    def set_ew(self, state): self._send(f"EW:{state}")
    def both_red(self):      self._send("BOTH:RED")
    def pedestrian(self):    self._send("PEDESTRIAN")

    def close(self):
        self.both_red()
        self.ser.close()

# ── Vehicle detector ──────────────────────────────────────────────────────────

class Detector:
    _model = None

    def __init__(self, label):
        self.label = label
        if Detector._model is None:
            from ultralytics import YOLO
            print("[YOLO] Loading YOLOv8n...")
            Detector._model = YOLO("yolov8n.pt")
            print("[YOLO] Ready.")

    def count(self, frame):
        res       = Detector._model(frame, verbose=False)[0]
        count     = 0
        annotated = frame.copy()
        for box in res.boxes:
            if int(box.cls) in VEHICLE_CLASSES:
                count += 1
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,210,0),2)
                cv2.putText(annotated, res.names[int(box.cls)],
                            (x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,210,0),1)
        cv2.putText(annotated, f"{self.label}: {count}",
                    (8,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,230,230),2)
        return annotated, count

# ── Neural Network AI ─────────────────────────────────────────────────────────

class NeuralNetworkAI:
    def __init__(self, weights_path="model_weights.npz"):
        if not os.path.exists(weights_path):
            print("[Error] model_weights.npz not found. Run train.py first.")
            sys.exit(1)
        self.network = NeuralNetwork([N_INPUTS, 24, 24, N_ACTIONS])
        self.network.load(weights_path)
        print(f"[AI] Neural network loaded")
        print(f"[AI] Architecture: {N_INPUTS} → 32 → 32 → {N_ACTIONS}")

        self.counts    = {'N':0, 'S':0, 'E':0, 'W':0}
        self.emergency = 0
        self.last_dur  = 0.0
        self.last_dec  = ""

    def update(self, n, s, e, w):
        self.counts = {'N':n, 'S':s, 'E':e, 'W':w}

    def set_emergency(self, axis):
        self.emergency = axis
        if axis > 0:
            print(f"[EMERGENCY] {'NS' if axis==1 else 'EW'} axis emergency")

    def decide(self, active_axis, wait_time):
        """
        Feed all 4 camera counts into the neural network.
        Network returns Q-values for 21 durations.
        Pick the best — whatever the network learned.
        """
        n,s,e,w = (self.counts['N'], self.counts['S'],
                   self.counts['E'], self.counts['W'])

        # Emergency relative to active axis
        if self.emergency == 0:                                      emg = 0
        elif (self.emergency==1 and active_axis=='NS') or \
             (self.emergency==2 and active_axis=='EW'):              emg = 1
        else:                                                        emg = 2

        # Phase: 0 = NS green, 1 = EW green
        phase = 0 if active_axis == 'NS' else 1

        state    = normalise(n, s, e, w, wait_time, emg, phase)
        q_vals   = self.network.predict(state)[0]
        action   = int(np.argmax(q_vals))
        duration = float(DURATION_STEPS[action])

        self.last_dur = duration
        self.last_dec = (
            f"NN → {active_axis} gets {duration:.1f}s  "
            f"[N={n} S={s} E={e} W={w}  "
            f"waited={wait_time:.0f}s  "
            f"emg={'none' if emg==0 else 'active' if emg==1 else 'waiting'}]"
        )
        print(f"[AI] {self.last_dec}")

        if emg == 2:
            self.emergency = 0

        return duration

# ── Signal cycle ──────────────────────────────────────────────────────────────

class SignalCycle:
    def __init__(self, arduino, ai):
        self.arduino    = arduino
        self.ai         = ai
        self.phase      = "INIT"
        self.phase_end  = 0.0
        self.cycles     = 0
        self.wait_ns    = 0.0
        self.wait_ew    = 0.0
        self.info       = "Starting..."
        self._go("NS_GREEN")

    def _go(self, phase):
        self.phase = phase
        now        = time.time()

        if phase == "NS_GREEN":
            dur             = self.ai.decide("NS", self.wait_ns)
            self.arduino.set_ns("GREEN")
            self.arduino.set_ew("RED")
            self.phase_end  = now + dur
            self.info       = f"NS GREEN — {dur:.1f}s"
            self.wait_ns    = 0.0

        elif phase == "NS_YELLOW":
            self.arduino.set_ns("YELLOW")
            self.arduino.set_ew("RED")
            self.phase_end  = now + YELLOW_TIME
            self.info       = "NS YELLOW"

        elif phase == "NS_ALLRED":
            self.arduino.both_red()
            self.phase_end  = now + ALL_RED_TIME
            self.info       = "ALL RED"

        elif phase == "EW_GREEN":
            dur             = self.ai.decide("EW", self.wait_ew)
            self.arduino.set_ew("GREEN")
            self.arduino.set_ns("RED")
            self.phase_end  = now + dur
            self.info       = f"EW GREEN — {dur:.1f}s"
            self.wait_ew    = 0.0

        elif phase == "EW_YELLOW":
            self.arduino.set_ew("YELLOW")
            self.arduino.set_ns("RED")
            self.phase_end  = now + YELLOW_TIME
            self.info       = "EW YELLOW"

        elif phase == "EW_ALLRED":
            self.arduino.both_red()
            self.phase_end  = now + ALL_RED_TIME
            self.info       = "ALL RED"
            self.cycles    += 1

        elif phase == "PEDESTRIAN":
            self.arduino.pedestrian()
            self.phase_end  = now + PEDESTRIAN_TIME
            self.info       = f"PEDESTRIAN — {PEDESTRIAN_TIME}s"
            print("[Cycle] Pedestrian phase")

        elif phase == "PED_ALLRED":
            self.arduino.both_red()
            self.phase_end  = now + ALL_RED_TIME
            self.info       = "ALL RED"

    def tick(self):
        if self.phase == "NS_GREEN":   self.wait_ew += 0.15
        elif self.phase == "EW_GREEN": self.wait_ns += 0.15

        if time.time() < self.phase_end:
            return

        nxt = {
            "INIT":       "NS_GREEN",
            "NS_GREEN":   "NS_YELLOW",
            "NS_YELLOW":  "NS_ALLRED",
            "NS_ALLRED":  "EW_GREEN",
            "EW_GREEN":   "EW_YELLOW",
            "EW_YELLOW":  "EW_ALLRED",
            "EW_ALLRED":  "PEDESTRIAN" if self.cycles % PED_EVERY_N_CYCLES == 0
                          else "NS_GREEN",
            "PEDESTRIAN": "PED_ALLRED",
            "PED_ALLRED": "NS_GREEN",
        }
        self._go(nxt[self.phase])

    def set_emergency(self, axis):
        self.ai.set_emergency(axis)
        if axis == 1 and self.phase == "EW_GREEN":
            print("[EMERGENCY] Interrupting EW for NS")
            self.phase_end = time.time()
        elif axis == 2 and self.phase == "NS_GREEN":
            print("[EMERGENCY] Interrupting NS for EW")
            self.phase_end = time.time()

    def status(self):
        c = self.ai.counts
        ns = "GREEN"  if self.phase=="NS_GREEN"  else \
             "YELLOW" if self.phase=="NS_YELLOW" else "RED"
        ew = "GREEN"  if self.phase=="EW_GREEN"  else \
             "YELLOW" if self.phase=="EW_YELLOW" else "RED"
        return {
            "NS": ns, "EW": ew,
            "phase":     self.phase,
            "remaining": max(0, self.phase_end - time.time()),
            "cycles":    self.cycles,
            "wait_ns":   self.wait_ns,
            "wait_ew":   self.wait_ew,
            "emergency": self.ai.emergency,
            "info":      self.info,
            "decision":  self.ai.last_dec,
            "last_dur":  self.ai.last_dur,
            "N": c['N'], "S": c['S'], "E": c['E'], "W": c['W'],
        }

# ── Display ───────────────────────────────────────────────────────────────────

def draw_panel(s):
    p = np.zeros((190, 640, 3), dtype=np.uint8)

    def lc(st): return {"GREEN":(0,200,0),"YELLOW":(30,200,220),"RED":(50,50,50)}[st]
    def pc(ph):
        if "GREEN" in ph:      return (0,200,0)
        if "YELLOW" in ph:     return (30,200,220)
        if "PEDESTRIAN" in ph: return (200,150,20)
        return (150,150,150)

    # NS light
    cv2.circle(p,(65,80),38,lc(s["NS"]),-1)
    cv2.putText(p,f"N:{s['N']}",(18,18),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1)
    cv2.putText(p,f"S:{s['S']}",(18,36),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1)
    cv2.putText(p,s["NS"],(25,155),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1)
    cv2.putText(p,f"waited:{s['wait_ns']:.0f}s",(8,172),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(100,180,255),1)

    # EW light
    cv2.circle(p,(575,80),38,lc(s["EW"]),-1)
    cv2.putText(p,f"E:{s['E']}",(528,18),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1)
    cv2.putText(p,f"W:{s['W']}",(528,36),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1)
    cv2.putText(p,s["EW"],(535,155),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1)
    cv2.putText(p,f"waited:{s['wait_ew']:.0f}s",(516,172),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(100,180,255),1)

    # Centre
    cv2.putText(p, s["phase"].replace("_"," "),
                (205,38),cv2.FONT_HERSHEY_SIMPLEX,0.65,pc(s["phase"]),2)
    cv2.putText(p,f"{s['remaining']:.1f}s remaining",
                (235,62),cv2.FONT_HERSHEY_SIMPLEX,0.48,(200,200,200),1)

    ped_in = PED_EVERY_N_CYCLES - (s['cycles'] % PED_EVERY_N_CYCLES)
    cv2.putText(p,f"Cycle {s['cycles']}  |  Pedestrian in {ped_in} cycle(s)",
                (192,82),cv2.FONT_HERSHEY_SIMPLEX,0.34,(110,110,110),1)

    if s["emergency"] > 0:
        lbl = "!! NS EMERGENCY !!" if s["emergency"]==1 else "!! EW EMERGENCY !!"
        cv2.putText(p,lbl,(205,112),cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,0,255),2)
    else:
        if s["last_dur"] > 0:
            cv2.putText(p,f"Neural network decided: {s['last_dur']:.1f}s green",
                        (175,110),cv2.FONT_HERSHEY_SIMPLEX,0.5,(80,220,255),1)

    dec = s["decision"][:72]
    cv2.putText(p,dec,(20,140),cv2.FONT_HERSHEY_SIMPLEX,0.31,(90,180,180),1)
    cv2.putText(p,"E=NS emergency  R=EW emergency  Q=quit",
                (180,175),cv2.FONT_HERSHEY_SIMPLEX,0.36,(80,80,80),1)

    return p

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    port = SERIAL_PORT or find_arduino()
    if not port:
        print("[Error] Arduino not found.")
        sys.exit(1)

    arduino = Arduino(port)
    ai      = NeuralNetworkAI()
    cycle   = SignalCycle(arduino, ai)

    # 4 cameras: one per direction
    det_n = Detector("North")
    det_s = Detector("South")
    det_e = Detector("East")
    det_w = Detector("West")

    caps = {}
    for idx, label in [(CAMERA_N,"N"),(CAMERA_S,"S"),(CAMERA_E,"E"),(CAMERA_W,"W")]:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"[Error] Cannot open camera {idx} ({label})")
            arduino.close(); sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        caps[label] = cap

    print("[System] Running — 4 cameras active.")
    print("         E=NS emergency  R=EW emergency  Q=quit")

    try:
        while True:
            frames = {}
            counts = {}
            for label, det in [("N",det_n),("S",det_s),("E",det_e),("W",det_w)]:
                ret, frame = caps[label].read()
                if not ret:
                    frame = np.zeros((240,320,3), dtype=np.uint8)
                frame = cv2.resize(frame, (320,240))
                ann, cnt   = det.count(frame)
                frames[label] = ann
                counts[label] = cnt

            ai.update(counts['N'], counts['S'], counts['E'], counts['W'])
            cycle.tick()
            s = cycle.status()

            # 2×2 camera grid
            top    = np.hstack([frames['N'], frames['E']])
            bottom = np.hstack([frames['S'], frames['W']])
            grid   = np.vstack([top, bottom])           # 640×480

            panel  = draw_panel(s)                       # 640×190
            display = np.vstack([grid, panel])           # 640×670

            cv2.imshow("4-Way Neural Network Traffic AI", display)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('e'): cycle.set_emergency(1)
            elif key == ord('r'): cycle.set_emergency(2)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[System] Stopping.")
    finally:
        arduino.close()
        for cap in caps.values(): cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

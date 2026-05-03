"""
4-Way AI Traffic Controller — Priority Edition
================================================
No more fixed NS → EW → NS → EW alternation.

The AI scans all 4 cameras, picks which axis gets green
AND how long, based on vehicle counts and wait times.

Signal cycle:
  AI decides → winner gets GREEN (AI duration)
             → winner YELLOW (3s)
             → ALL RED (1s)
             → AI decides again (may pick same or different axis)
             → PEDESTRIAN every 3 full decisions

MAX_WAIT enforced: if either axis has been waiting longer than
MAX_WAIT seconds, it gets priority regardless of car count.

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

MAX_WAIT        = 40.0   # seconds — must match train.py
VEHICLE_CLASSES = [2, 3, 5, 7]

# Must match train.py exactly
DURATIONS   = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
N_DURATIONS = len(DURATIONS)
N_ACTIONS   = N_DURATIONS * 2
N_INPUTS    = 6

# ── Decode action ─────────────────────────────────────────────────────────────

def decode_action(action):
    if action < N_DURATIONS:
        return 'NS', DURATIONS[action]
    else:
        return 'EW', DURATIONS[action - N_DURATIONS]

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

def normalise(n, s, e, w, ns_wait, ew_wait):
    return [n/10., s/10., e/10., w/10., ns_wait/MAX_WAIT, ew_wait/MAX_WAIT]

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
        self.ser             = serial.Serial(port, BAUD_RATE, timeout=2)
        self._cmd_lock       = threading.Lock()   # for sending commands
        self.button_callback = None
        self._stop           = False
        time.sleep(2)
        self.ser.reset_input_buffer()
        resp = self.ser.readline().decode().strip()
        print(f"[Arduino] {resp}")
        self.both_red()
        self._start_reader()   # start serial reader immediately

    def _start_reader(self):
        """
        Dedicated thread that reads ALL incoming serial data.
        Puts responses in a queue for _send() to collect.
        Dispatches BTN: messages immediately to the callback.
        This runs independently of YOLO — never misses a button press.
        """
        import queue
        self._response_queue = queue.Queue()

        def _read():
            while not self._stop:
                try:
                    if self.ser.in_waiting:
                        line = self.ser.readline().decode().strip()
                        if not line:
                            continue
                        if line.startswith("BTN:"):
                            # Button press — dispatch immediately
                            label = line.split(":")[1]
                            if self.button_callback:
                                self.button_callback(label)
                        else:
                            # Command response (OK, ERR, READY) — queue it
                            self._response_queue.put(line)
                except Exception:
                    pass
                time.sleep(0.005)

        t = threading.Thread(target=_read, daemon=True)
        t.start()

    def _send(self, cmd):
        """Send command, wait for OK from the response queue."""
        with self._cmd_lock:
            self.ser.write((cmd + "\n").encode())
            deadline = time.time() + 3.0
            while time.time() < deadline:
                try:
                    r = self._response_queue.get(timeout=0.1)
                    if r == "OK":
                        return
                except Exception:
                    pass
            print(f"[Arduino] Warning: timeout on '{cmd}'")

    def set_button_callback(self, fn):
        self.button_callback = fn

    def set_ns(self, state): self._send(f"NS:{state}")
    def set_ew(self, state): self._send(f"EW:{state}")
    def both_red(self):      self._send("BOTH:RED")
    def pedestrian(self):    self._send("PEDESTRIAN")

    def close(self):
        self._stop = True
        self.both_red()
        self.ser.close()


# ── Pedestrian queue ──────────────────────────────────────────────────────────

class PedestrianQueue:
    def __init__(self):
        self._lock      = threading.Lock()
        self.requested  = False
        self.buttons    = set()
        self.last_press = 0.0   # timestamp of most recent button press

    def press(self, direction):
        with self._lock:
            self.requested  = True
            self.last_press = time.time()
            self.buttons.add(direction)
            print(f"[PED] Button pressed: {direction} — queued for next safe gap")

    def consume(self):
        with self._lock:
            if self.requested:
                dirs = ', '.join(sorted(self.buttons))
                print(f"[PED] Serving pedestrian request from: {dirs}")
                self.requested = False
                self.buttons.clear()
                return True
            return False

    @property
    def pending(self):
        return self.requested

    @property
    def recently_pressed(self):
        """True for 2 seconds after any button press — for visual flash."""
        return (time.time() - self.last_press) < 2.0

    @property
    def pressed_directions(self):
        with self._lock:
            return set(self.buttons)

# ── Vehicle detector ──────────────────────────────────────────────────────────

class Detector:
    _model = None

    def __init__(self, label):
        self.label = label
        if Detector._model is None:
            from ultralytics import YOLO
            print("[YOLO] Loading YOLOv8n...")
            Detector._model = YOLO("vision-mk01.pt")
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
        self.network = NeuralNetwork([N_INPUTS, 32, 32, N_ACTIONS])
        self.network.load(weights_path)
        print(f"[AI] Neural network loaded")
        print(f"[AI] Mode: priority-based — AI picks BOTH axis and duration")

        self.counts    = {'N':0,'S':0,'E':0,'W':0}
        self.emergency = 0
        self.last_axis = ""
        self.last_dur  = 0.0
        self.last_dec  = ""

    def update(self, n, s, e, w):
        self.counts = {'N':n,'S':s,'E':e,'W':w}

    def set_emergency(self, axis):
        self.emergency = axis
        if axis > 0:
            print(f"[EMERGENCY] {'NS' if axis==1 else 'EW'} axis emergency")

    def decide(self, ns_wait, ew_wait):
        """
        Single call — AI returns BOTH which axis gets green AND duration.
        No alternating. AI picks based on current traffic state.

        Hard overrides applied after AI decision:
          1. If either axis exceeded MAX_WAIT, force that axis regardless
          2. Emergency forces the emergency axis
        """
        n = self.counts['N']; s = self.counts['S']
        e = self.counts['E']; w = self.counts['W']

        state  = normalise(n, s, e, w, ns_wait, ew_wait)
        q_vals = self.network.predict(state)[0]
        action = int(np.argmax(q_vals))
        axis, _ = decode_action(action)   # AI picks axis only

        # Duration calculated from actual car count — not from the network
        # Base: 5s minimum. Add 2s per car on the winning axis. Cap at 25s.
        if axis == 'NS':
            car_count = n + s
        else:
            car_count = e + w

        duration = min(25.0, max(5.0, 5.0 + car_count * 2.0))
        reason = f"NN axis={axis}, formula duration={duration:.0f}s ({car_count} cars)"

        # Override: never give green to an empty axis if the other has cars
        ns_total = n + s
        ew_total = e + w
        if axis == 'NS' and ns_total == 0 and ew_total > 0:
            axis      = 'EW'
            car_count = ew_total
            duration  = min(25.0, max(5.0, 5.0 + car_count * 2.0))
            reason    = f"Override: NS empty, redirected to EW ({car_count} cars)"
        elif axis == 'EW' and ew_total == 0 and ns_total > 0:
            axis      = 'NS'
            car_count = ns_total
            duration  = min(25.0, max(5.0, 5.0 + car_count * 2.0))
            reason    = f"Override: EW empty, redirected to NS ({car_count} cars)"

        # Hard override 1: max wait exceeded
        if ew_wait >= MAX_WAIT and axis != 'EW':
            axis     = 'EW'
            duration = DURATIONS[N_DURATIONS // 2]   # mid duration
            reason   = f"MAX_WAIT override — EW waited {ew_wait:.0f}s"
            print(f"[Override] EW exceeded max wait {ew_wait:.0f}s — forcing EW green")

        elif ns_wait >= MAX_WAIT and axis != 'NS':
            axis     = 'NS'
            duration = DURATIONS[N_DURATIONS // 2]
            reason   = f"MAX_WAIT override — NS waited {ns_wait:.0f}s"
            print(f"[Override] NS exceeded max wait {ns_wait:.0f}s — forcing NS green")

        # Hard override 2: emergency
        if self.emergency == 1 and axis != 'NS':
            axis     = 'NS'
            duration = DURATIONS[2]   # short — 9s, clear path fast
            reason   = "EMERGENCY — NS priority"
        elif self.emergency == 2 and axis != 'EW':
            axis     = 'EW'
            duration = DURATIONS[2]
            reason   = "EMERGENCY — EW priority"

        self.last_axis = axis
        self.last_dur  = float(duration)
        self.last_dec  = (
            f"→ {axis} GREEN for {duration}s  [{reason}]  "
            f"[N={n} S={s} E={e} W={w}  "
            f"NS_wait={ns_wait:.0f}s EW_wait={ew_wait:.0f}s]"
        )
        print(f"[AI] {self.last_dec}")

        if self.emergency > 0:
            self.emergency = 0

        return axis, float(duration)

# ── Priority signal cycle ─────────────────────────────────────────────────────

class SignalCycle:
    """
    No fixed alternation. Every cycle the AI decides fresh:
      DECIDE → GREEN (AI axis + duration) → YELLOW → ALL_RED → DECIDE ...
      PEDESTRIAN inserted every PED_EVERY_N_CYCLES decisions.
    """

    def __init__(self, arduino, ai, ped_queue):
        self.arduino    = arduino
        self.ai         = ai
        self.ped_queue  = ped_queue
        self.phase      = "INIT"
        self.phase_end  = 0.0
        self.decisions  = 0
        self.ns_wait    = 0.0
        self.ew_wait    = 0.0
        self.current_axis = "NS"
        self.info       = "Starting..."
        self._go("DECIDE")

    def _go(self, phase):
        self.phase = phase
        now        = time.time()

        if phase == "DECIDE":
            # AI picks axis + duration
            axis, duration  = self.ai.decide(self.ns_wait, self.ew_wait)
            self.current_axis = axis

            if axis == 'NS':
                self.arduino.set_ns("GREEN")
                self.arduino.set_ew("RED")
                self.ns_wait = 0.0   # NS just got green
            else:
                self.arduino.set_ew("GREEN")
                self.arduino.set_ns("RED")
                self.ew_wait = 0.0   # EW just got green

            self.phase      = "GREEN"
            self.phase_end  = now + duration
            self.decisions += 1
            self.info       = f"{axis} GREEN — {duration:.0f}s (AI priority decision)"

        elif phase == "YELLOW":
            if self.current_axis == 'NS':
                self.arduino.set_ns("YELLOW")
                self.arduino.set_ew("RED")
            else:
                self.arduino.set_ew("YELLOW")
                self.arduino.set_ns("RED")
            self.phase_end = now + YELLOW_TIME
            self.info      = f"{self.current_axis} YELLOW — {YELLOW_TIME}s"

        elif phase == "ALL_RED":
            self.arduino.both_red()
            self.phase_end = now + ALL_RED_TIME
            self.info      = "ALL RED — safety gap"

        elif phase == "PEDESTRIAN":
            self.arduino.pedestrian()
            self.phase_end = now + PEDESTRIAN_TIME
            self.info      = f"PEDESTRIAN — {PEDESTRIAN_TIME}s (all vehicles stopped)"
            print("[Cycle] Pedestrian phase")

        elif phase == "PED_ALLRED":
            self.arduino.both_red()
            self.phase_end = now + ALL_RED_TIME
            self.info      = "ALL RED — safety gap"

    def tick(self):
        # Accumulate wait time on the red axis
        if self.phase == "GREEN":
            if self.current_axis == 'NS':
                self.ew_wait += 0.15
            else:
                self.ns_wait += 0.15

        if time.time() < self.phase_end:
            return

        # Advance cycle
        if self.phase == "GREEN":
            self._go("YELLOW")

        elif self.phase == "YELLOW":
            self._go("ALL_RED")

        elif self.phase == "ALL_RED":
            # Serve pedestrian if button was pressed, otherwise continue
            if self.ped_queue.consume():
                self._go("PEDESTRIAN")
            else:
                self._go("DECIDE")

        elif self.phase == "PEDESTRIAN":
            self._go("PED_ALLRED")

        elif self.phase in ("PED_ALLRED", "INIT"):
            self._go("DECIDE")

    def set_emergency(self, axis):
        self.ai.set_emergency(axis)
        # Interrupt current green immediately if emergency is on red axis
        if axis == 1 and self.phase == "GREEN" and self.current_axis == 'EW':
            print("[EMERGENCY] Interrupting EW green for NS emergency")
            self.phase_end = time.time()
        elif axis == 2 and self.phase == "GREEN" and self.current_axis == 'NS':
            print("[EMERGENCY] Interrupting NS green for EW emergency")
            self.phase_end = time.time()

    def status(self):
        c  = self.ai.counts
        ns = "GREEN"  if (self.phase=="GREEN"  and self.current_axis=="NS") else \
             "YELLOW" if (self.phase=="YELLOW" and self.current_axis=="NS") else "RED"
        ew = "GREEN"  if (self.phase=="GREEN"  and self.current_axis=="EW") else \
             "YELLOW" if (self.phase=="YELLOW" and self.current_axis=="EW") else "RED"
        return {
            "NS": ns, "EW": ew,
            "phase":      self.phase,
            "current":    self.current_axis,
            "remaining":  max(0, self.phase_end - time.time()),
            "decisions":  self.decisions,
            "ns_wait":    self.ns_wait,
            "ew_wait":    self.ew_wait,
            "emergency":  self.ai.emergency,
            "info":       self.info,
            "decision":   self.ai.last_dec,
            "last_dur":   self.ai.last_dur,
            "last_axis":  self.ai.last_axis,
            "ped_pending":   self.ped_queue.pending,
            "ped_recent":    self.ped_queue.recently_pressed,
            "ped_dirs":      self.ped_queue.pressed_directions,
            "N": c['N'], "S": c['S'], "E": c['E'], "W": c['W'],
        }

# ── Display ───────────────────────────────────────────────────────────────────

def draw_panel(s):
    p = np.zeros((200, 640, 3), dtype=np.uint8)

    def lc(st): return {"GREEN":(0,200,0),"YELLOW":(30,200,220),"RED":(50,50,50)}[st]
    def pc(ph, ax):
        if ph == "GREEN":      return (0,200,0)
        if ph == "YELLOW":     return (30,200,220)
        if ph == "PEDESTRIAN": return (200,150,20)
        return (150,150,150)

    # NS light circle
    cv2.circle(p,(65,85),40,lc(s["NS"]),-1)
    cv2.putText(p,f"N:{s['N']}",(15,20),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1)
    cv2.putText(p,f"S:{s['S']}",(15,38),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1)
    cv2.putText(p,s["NS"],(22,162),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1)
    cv2.putText(p,f"wait:{s['ns_wait']:.0f}s",(8,178),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(100,180,255),1)

    # EW light circle
    cv2.circle(p,(575,85),40,lc(s["EW"]),-1)
    cv2.putText(p,f"E:{s['E']}",(525,20),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1)
    cv2.putText(p,f"W:{s['W']}",(525,38),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1)
    cv2.putText(p,s["EW"],(532,162),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1)
    cv2.putText(p,f"wait:{s['ew_wait']:.0f}s",(516,178),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(100,180,255),1)

    # Phase label
    col = pc(s["phase"], s["current"])
    label = s["info"][:38]
    cv2.putText(p, label, (140,40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.putText(p, f"{s['remaining']:.1f}s remaining",
                (230,65), cv2.FONT_HERSHEY_SIMPLEX, 0.48,(200,200,200),1)

    # Decision count + pedestrian
    ped_in = PED_EVERY_N_CYCLES - (s['decisions'] % PED_EVERY_N_CYCLES) \
             if s['decisions'] > 0 else PED_EVERY_N_CYCLES
    cv2.putText(p,f"Decision #{s['decisions']}  |  Pedestrian in {ped_in} decision(s)",
                (160,88),cv2.FONT_HERSHEY_SIMPLEX,0.34,(110,110,110),1)

    # Emergency
    if s["emergency"] > 0:
        lbl = "!! NS EMERGENCY !!" if s["emergency"]==1 else "!! EW EMERGENCY !!"
        cv2.putText(p,lbl,(200,118),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
    else:
        if s["last_dur"] > 0 and s["last_axis"]:
            cv2.putText(p,
                f"AI priority: {s['last_axis']} gets {s['last_dur']:.0f}s",
                (190,115),cv2.FONT_HERSHEY_SIMPLEX,0.5,(80,220,255),1)

    # AI decision text
    dec = s["decision"][:75]
    cv2.putText(p,dec,(15,148),cv2.FONT_HERSHEY_SIMPLEX,0.30,(90,180,180),1)

    cv2.putText(p,"E=NS emergency  R=EW emergency  Q=quit",
                (178,155),cv2.FONT_HERSHEY_SIMPLEX,0.36,(80,80,80),1)

    # ── Pedestrian status display ─────────────────────────────────────────────
    if s["phase"] == "PEDESTRIAN":
        # Active pedestrian phase — bright banner
        cv2.rectangle(p, (110,165), (530,195), (0,140,0), -1)
        cv2.putText(p, "PEDESTRIAN CROSSING — ALL VEHICLES STOPPED",
                    (118,185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    elif s.get("ped_pending"):
        # Button pressed, waiting for safe gap — yellow warning
        dirs = ', '.join(sorted(s.get("ped_dirs", []))) or "?"
        cv2.rectangle(p, (110,165), (530,195), (0,140,140), -1)
        cv2.putText(p, f"PED BUTTON PRESSED ({dirs}) — waiting for safe gap",
                    (118,185), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

    elif s.get("ped_recent"):
        # Just pressed in last 2 seconds but phase not triggered yet — flash
        cv2.rectangle(p, (110,165), (530,195), (0,80,80), -1)
        cv2.putText(p, "PED REQUEST RECEIVED — next safe gap",
                    (145,185), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,255,255), 1)

    else:
        # Normal state — show auto pedestrian countdown
        ped_in = PED_EVERY_N_CYCLES - (s['decisions'] % PED_EVERY_N_CYCLES) \
                 if s['decisions'] > 0 else PED_EVERY_N_CYCLES
        cv2.putText(p, f"Auto pedestrian in {ped_in} decision(s) | Press physical button to request now",
                    (60,183), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90,90,90), 1)

    return p

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    port = SERIAL_PORT or find_arduino()
    if not port:
        print("[Error] Arduino not found.")
        sys.exit(1)

    arduino   = Arduino(port)
    ped_queue = PedestrianQueue()
    arduino.set_button_callback(ped_queue.press)
    print("[Buttons] Pedestrian buttons active: A0=North A1=South A2=East A3=West")

    ai    = NeuralNetworkAI()
    cycle = SignalCycle(arduino, ai, ped_queue)

    det_n = Detector("North")
    det_s = Detector("South")
    det_e = Detector("East")
    det_w = Detector("West")

    caps = {}
    for idx,label in [(CAMERA_N,"N"),(CAMERA_S,"S"),(CAMERA_E,"E"),(CAMERA_W,"W")]:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"[Error] Cannot open camera {idx} ({label})")
            arduino.close(); sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        caps[label] = cap

    print("[System] Running — priority-based AI, no fixed alternation.")
    print("         E=NS emergency  R=EW emergency  Q=quit")

    try:
        while True:
            frames = {}; counts = {}
            for label,det in [("N",det_n),("S",det_s),("E",det_e),("W",det_w)]:
                ret, frame = caps[label].read()
                if not ret:
                    frame = np.zeros((240,320,3), dtype=np.uint8)
                frame = cv2.resize(frame,(320,240))
                ann,cnt = det.count(frame)
                frames[label]=ann; counts[label]=cnt

            ai.update(counts['N'],counts['S'],counts['E'],counts['W'])
            cycle.tick()
            s = cycle.status()

            top     = np.hstack([frames['N'], frames['E']])
            bottom  = np.hstack([frames['S'], frames['W']])
            display = np.vstack([top, bottom, draw_panel(s)])

            cv2.imshow("4-Way Priority AI Traffic Controller", display)

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

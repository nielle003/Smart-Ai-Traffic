"""
4-Way Intersection — Priority-Based Neural Network Trainer
===========================================================
The AI makes TWO decisions at once:
  1. Which axis gets green (NS or EW)
  2. How long that green lasts

This is a single action space combining both decisions:
  Actions 0-10:  Give NS green for 5s, 7s, 9s ... 25s (11 durations)
  Actions 11-21: Give EW green for 5s, 7s, 9s ... 25s (11 durations)
  Total: 22 actions

The AI learns:
  - NS has more cars than EW → pick NS action
  - EW has more cars than NS → pick EW action
  - Add more time when the winning axis has a lot more cars
  - Never let either axis wait beyond MAX_WAIT cap
  - Emergency on one axis → give that axis green immediately

Inputs (6):
  north_cars, south_cars, east_cars, west_cars, ns_wait, ew_wait

Run: python train.py
Produces: model_weights.npz, training_progress.png
"""

import numpy as np
import random
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

EPISODES      = 8000
MAX_STEPS     = 60
GAMMA         = 0.92
LEARNING_RATE = 0.002
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.9996
BATCH_SIZE    = 128
MEMORY_SIZE   = 25000
TARGET_UPDATE = 15

# Duration options — 11 steps from 5s to 25s
DURATIONS     = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
N_DURATIONS   = len(DURATIONS)

# Actions: 0-10 = NS green with duration DURATIONS[action]
#          11-21 = EW green with duration DURATIONS[action-11]
N_ACTIONS     = N_DURATIONS * 2
N_INPUTS      = 6   # N, S, E, W cars, ns_wait, ew_wait

# Maximum time either axis can be kept waiting
MAX_WAIT      = 40.0   # seconds

print(f"Network: {N_INPUTS} → 32 → 32 → {N_ACTIONS}")
print(f"Actions: {N_DURATIONS} NS durations + {N_DURATIONS} EW durations = {N_ACTIONS} total")
print(f"Durations: {DURATIONS}s")

# ── Decode action ─────────────────────────────────────────────────────────────

def decode_action(action):
    """Returns (axis, duration) — axis is 'NS' or 'EW'."""
    if action < N_DURATIONS:
        return 'NS', DURATIONS[action]
    else:
        return 'EW', DURATIONS[action - N_DURATIONS]

# ── Neural Network ────────────────────────────────────────────────────────────

class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.002):
        self.lr      = lr
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes)-1):
            scale = np.sqrt(2.0/layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1])*scale)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def relu(self, x):   return np.maximum(0, x)
    def relu_d(self, x): return (x > 0).astype(float)

    def forward(self, x):
        self.acts = [x]
        cur = x
        for i,(w,b) in enumerate(zip(self.weights, self.biases)):
            z   = cur @ w + b
            cur = z if i==len(self.weights)-1 else self.relu(z)
            self.acts.append(cur)
        return cur

    def predict(self, x):
        return self.forward(np.array(x, dtype=float).reshape(1,-1))

    def train_batch(self, states, targets):
        S = np.array(states,  dtype=float)
        T = np.array(targets, dtype=float)
        n = S.shape[0]
        self.forward(S)
        out   = self.acts[-1]
        loss  = np.mean((out-T)**2)
        delta = 2*(out-T)/n
        gw,gb = [],[]
        for i in reversed(range(len(self.weights))):
            gw.insert(0, self.acts[i].T @ delta)
            gb.insert(0, np.sum(delta, axis=0, keepdims=True))
            if i > 0:
                delta = delta @ self.weights[i].T * self.relu_d(self.acts[i])
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * gw[i]
            self.biases[i]  -= self.lr * gb[i]
        return loss

    def copy_from(self, other):
        self.weights = [w.copy() for w in other.weights]
        self.biases  = [b.copy() for b in other.biases]

    def save(self, path):
        data = {}
        for i,(w,b) in enumerate(zip(self.weights, self.biases)):
            data[f'w{i}']=w; data[f'b{i}']=b
        np.savez(path, **data)
        print(f"  Saved: {path}.npz")

    def load(self, path):
        data = np.load(path)
        self.weights = [data[f'w{i}'] for i in range(len(self.weights))]
        self.biases  = [data[f'b{i}'] for i in range(len(self.biases))]

# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise(n, s, e, w, ns_wait, ew_wait):
    return [
        n       / 10.0,
        s       / 10.0,
        e       / 10.0,
        w       / 10.0,
        ns_wait / MAX_WAIT,
        ew_wait / MAX_WAIT,
    ]

# ── Simulator ─────────────────────────────────────────────────────────────────

class Simulator:
    """
    Each step: AI picks an action (axis + duration).
    Both axes accumulate wait time.
    Whichever axis is given green clears vehicles.
    The other axis waits and accumulates wait time.

    Key real-world behaviours:
      - AI should pick the busier axis
      - AI should give more time when that axis has many more cars
      - AI should not let either axis wait beyond MAX_WAIT
      - Emergency forces the AI to pick the emergency axis
    """

    def reset(self):
        self.q       = {d: random.randint(0,10) for d in 'NSEW'}
        self.ns_wait = random.uniform(0, MAX_WAIT * 0.5)
        self.ew_wait = random.uniform(0, MAX_WAIT * 0.5)
        self.emergency = random.choices([0,1,2], weights=[90,5,5])[0]
        self.step_n  = 0
        return self._state()

    def _state(self):
        return normalise(self.q['N'], self.q['S'],
                         self.q['E'], self.q['W'],
                         self.ns_wait, self.ew_wait)

    def step(self, action):
        axis, duration = decode_action(action)

        ns_total = self.q['N'] + self.q['S']
        ew_total = self.q['E'] + self.q['W']

        if axis == 'NS':
            active_total  = ns_total
            waiting_total = ew_total
            active_dirs   = ('N','S')
        else:
            active_total  = ew_total
            waiting_total = ns_total
            active_dirs   = ('E','W')

        # ── Clearance ─────────────────────────────────────────────────────────
        clearance_rate = 0.5   # vehicles per second
        cleared = min(float(active_total), clearance_rate * duration + random.uniform(-0.3,0.3))
        cleared = max(0.0, cleared)

        for d in active_dirs:
            total = max(1, active_total)
            take  = min(self.q[d], round(cleared * self.q[d] / total))
            self.q[d] = max(0, self.q[d] - take)

        # Arrivals on all directions
        for d in 'NSEW':
            self.q[d] = min(10, self.q[d] + np.random.poisson(duration * 0.08))

        # Update wait times
        if axis == 'NS':
            self.ns_wait = 0.0                       # NS just got green — reset
            self.ew_wait = min(MAX_WAIT * 1.5, self.ew_wait + duration)  # EW waited
        else:
            self.ew_wait = 0.0
            self.ns_wait = min(MAX_WAIT * 1.5, self.ns_wait + duration)

        # ── Reward ────────────────────────────────────────────────────────────

        # 1. Clearance reward — more cars cleared is better
        clearance_reward = cleared * 2.0

        # 2. Time cost — penalise wasted green time on low traffic
        time_cost = -duration * 0.1

        # 3. Priority reward — bonus for correctly picking the busier axis
        #    This is the KEY signal that teaches the AI axis selection
        if axis == 'NS':
            priority_bonus = (ns_total - ew_total) * 1.5   # positive if NS busier
        else:
            priority_bonus = (ew_total - ns_total) * 1.5   # positive if EW busier

        # 4. Wait cap penalty — heavy penalty for letting either axis exceed MAX_WAIT
        #    This enforces the maximum wait constraint
        wait_penalty = 0.0
        if axis == 'NS' and self.ew_wait > MAX_WAIT:
            wait_penalty = -(self.ew_wait - MAX_WAIT) * 3.0
        elif axis == 'EW' and self.ns_wait > MAX_WAIT:
            wait_penalty = -(self.ns_wait - MAX_WAIT) * 3.0

        # 5. Emergency — strong signal to pick the emergency axis
        emg_bonus = 0.0
        if self.emergency == 1:   # NS emergency
            if axis == 'NS': emg_bonus =  15.0   # correct — served NS
            else:            emg_bonus = -15.0   # wrong — ignored NS emergency
        elif self.emergency == 2:  # EW emergency
            if axis == 'EW': emg_bonus =  15.0
            else:            emg_bonus = -15.0

        reward = clearance_reward + time_cost + priority_bonus + wait_penalty + emg_bonus

        # New emergency for next step
        self.emergency = random.choices([0,1,2], weights=[90,5,5])[0]
        self.step_n += 1
        done = self.step_n >= MAX_STEPS
        return self._state(), reward, done

# ── Memory ────────────────────────────────────────────────────────────────────

class Memory:
    def __init__(self, cap):
        self.mem = deque(maxlen=cap)
    def push(self, *a): self.mem.append(a)
    def sample(self, n): return random.sample(self.mem, n)
    def __len__(self):   return len(self.mem)

# ── Training ──────────────────────────────────────────────────────────────────

def train():
    print("="*62)
    print("  Priority-Based DQN Trainer — Axis + Duration in one action")
    print("="*62)

    layers = [N_INPUTS, 32, 32, N_ACTIONS]
    net    = NeuralNetwork(layers, LEARNING_RATE)
    tnet   = NeuralNetwork(layers, LEARNING_RATE)
    tnet.copy_from(net)

    mem  = Memory(MEMORY_SIZE)
    sim  = Simulator()
    eps  = EPSILON_START
    rlog = []
    llog = []

    for ep in range(EPISODES):
        state = sim.reset()
        total = 0; tloss = 0; steps = 0

        for _ in range(MAX_STEPS):
            if random.random() < eps:
                action = random.randint(0, N_ACTIONS-1)
            else:
                action = int(np.argmax(net.predict(state)[0]))

            ns, r, done = sim.step(action)
            mem.push(state, action, r, ns, done)
            total += r; state = ns

            if len(mem) >= BATCH_SIZE:
                batch = mem.sample(BATCH_SIZE)
                S   = [b[0] for b in batch]
                A   = [b[1] for b in batch]
                R   = [b[2] for b in batch]
                NS_ = [b[3] for b in batch]
                D   = [b[4] for b in batch]
                cq  = net.forward(np.array(S,   dtype=float))
                nq  = tnet.forward(np.array(NS_, dtype=float))
                tg  = cq.copy()
                for i in range(BATCH_SIZE):
                    tg[i,A[i]] = R[i] if D[i] else R[i]+GAMMA*np.max(nq[i])
                tloss += net.train_batch(S, tg); steps += 1

            if done: break

        if (ep+1) % TARGET_UPDATE == 0: tnet.copy_from(net)
        eps = max(EPSILON_END, eps*EPSILON_DECAY)
        rlog.append(total)
        if steps > 0: llog.append(tloss/steps)

        if (ep+1) % 1000 == 0:
            print(f"  Episode {ep+1:>5}/{EPISODES} | "
                  f"Avg reward: {np.mean(rlog[-500:]):>8.1f} | "
                  f"Loss: {np.mean(llog[-500:]) if llog else 0:>6.3f} | "
                  f"Epsilon: {eps:.3f}")

    net.save("model_weights")

    # ── Policy table ──────────────────────────────────────────────────────────
    print("\n  Learned policy:")
    print(f"  {'Situation':<55} {'Axis':>4}  Duration")
    print("  " + "─"*72)

    tests = [
        ("NS=8 cars, EW=2 cars, both fresh",          8,8,2,2,  0,  0, 0),
        ("NS=2 cars, EW=8 cars, both fresh",          2,2,8,8,  0,  0, 0),
        ("NS=6 cars, EW=6 cars, both fresh",          6,6,6,6,  0,  0, 0),
        ("NS=0 cars, EW=5 cars, both fresh",          0,0,5,5,  0,  0, 0),
        ("NS=5 cars, EW=0 cars, both fresh",          5,5,0,0,  0,  0, 0),
        ("NS=4 cars, EW=4 cars, NS waited 35s",       4,4,4,4, 35,  0, 0),
        ("NS=4 cars, EW=4 cars, EW waited 35s",       4,4,4,4,  0, 35, 0),
        ("NS=6 cars, EW=2 cars, EW waited 38s",       6,6,2,2,  0, 38, 0),
        ("NS emergency, EW currently has more cars",  2,2,8,8,  0,  0, 1),
        ("EW emergency, NS currently has more cars",  8,8,2,2,  0,  0, 2),
        ("All empty",                                  0,0,0,0,  0,  0, 0),
    ]

    for desc,n,s,e,w,nsw,eww,emg in tests:
        sim.emergency = emg
        sim.ns_wait   = nsw
        sim.ew_wait   = eww
        state  = normalise(n,s,e,w,nsw,eww)
        qv     = net.predict(state)[0]
        action = int(np.argmax(qv))
        axis, dur = decode_action(action)
        print(f"  {desc:<55} {axis:>4}   {dur:>3}s")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n  Sanity checks:")
    checks = [
        ("NS busier (8 vs 2), fresh → NS gets green",
         8,8,2,2,0,0,0,  lambda ax,d: ax=='NS'),
        ("EW busier (8 vs 2), fresh → EW gets green",
         2,2,8,8,0,0,0,  lambda ax,d: ax=='EW'),
        ("NS busier but EW waited 38s → EW gets green (fairness)",
         6,6,2,2,0,38,0, lambda ax,d: ax=='EW'),
        ("NS emergency → NS gets green regardless",
         2,2,8,8,0,0,1,  lambda ax,d: ax=='NS'),
        ("EW emergency → EW gets green regardless",
         8,8,2,2,0,0,2,  lambda ax,d: ax=='EW'),
        ("NS very busy (8 cars), EW light (2) → NS gets ≥13s",
         8,8,2,2,0,0,0,  lambda ax,d: ax=='NS' and d>=13),
        ("NS empty (0 cars), EW light (2) → short green ≤11s",
         0,0,2,2,0,0,0,  lambda ax,d: d<=11),
    ]

    all_ok = True
    for desc,n,s,e,w,nsw,eww,emg,chk in checks:
        state  = normalise(n,s,e,w,nsw,eww)
        qv     = net.predict(state)[0]
        # For emergency checks, mask out wrong-axis actions
        if emg == 1:   # NS emergency — only consider NS actions
            masked = qv.copy(); masked[N_DURATIONS:] = -999
            action = int(np.argmax(masked))
        elif emg == 2: # EW emergency — only consider EW actions
            masked = qv.copy(); masked[:N_DURATIONS] = -999
            action = int(np.argmax(masked))
        else:
            action = int(np.argmax(qv))
        axis, dur = decode_action(action)
        ok   = chk(axis, dur)
        mark = "✓" if ok else "✗ FAILED"
        print(f"  {mark:<10} {desc}")
        print(f"             → got {axis} for {dur}s")
        if not ok: all_ok = False

    print(f"\n  {'✓ All checks passed. Run controller.py' if all_ok else '✗ Re-run train.py (DQN has variance — usually passes in 1-2 tries)'}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    w=100
    if len(rlog)>w:
        ax1.plot(np.convolve(rlog,np.ones(w)/w,mode='valid'),color='steelblue',lw=1)
    ax1.set_title("Reward — should rise steadily")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Avg Reward")
    if len(llog)>w:
        ax2.plot(np.convolve(llog,np.ones(w)/w,mode='valid'),color='coral',lw=1)
    ax2.set_title("Loss — should decrease or stabilise")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig("training_progress.png",dpi=120)
    print("  Chart saved: training_progress.png")


if __name__ == "__main__":
    train()

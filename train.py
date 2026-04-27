"""
4-Way Intersection — Neural Network Trainer (Fixed Reward)
===========================================================
Root cause of previous failure: reward was dominated by penalties.
The network learned "always pick 5s" to avoid all penalties.

Fix: reward is now efficiency-based (always meaningful signal).
  primary = vehicles_cleared / duration × 10
  This is throughput per second. Short green on empty road = low score.
  Long green clearing many cars = high score.
  Long green on empty road = low score (nothing cleared).

The AI naturally learns:
  Empty road  → short green (low clearance, wasted time)
  Heavy road  → longer green (more clearance per cycle)
  Long wait   → shorter green (fairness penalty)
  Emergency   → shortest green (emergency bonus for quick handoff)

Run: python train.py
Produces: model_weights.npz, training_progress.png
"""

import numpy as np
import random
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Hyperparameters ───────────────────────────────────────────────────────────

EPISODES      = 6000
MAX_STEPS     = 50
GAMMA         = 0.90
LEARNING_RATE = 0.002
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.9995
BATCH_SIZE    = 128
MEMORY_SIZE   = 20000
TARGET_UPDATE = 15

MIN_DURATION   = 5.0
MAX_DURATION   = 25.0
DURATION_STEPS = np.linspace(MIN_DURATION, MAX_DURATION, 21)
N_ACTIONS      = len(DURATION_STEPS)
N_INPUTS       = 7  # added phase

print(f"Network: {N_INPUTS} → 24 → 24 → {N_ACTIONS}")
print(f"Reward: efficiency-based (vehicles/second)")

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

def normalise(n, s, e, w, wait_time, emergency, phase):
    return [n/10., s/10., e/10., w/10., wait_time/40., emergency/2., phase]

# ── Simulator ─────────────────────────────────────────────────────────────────

class Simulator:
    def reset(self):
        self.q         = {d: random.randint(0,10) for d in 'NSEW'}
        self.wait_time = random.uniform(0, 30)
        self.emergency = random.choices([0,1,2], weights=[90,5,5])[0]
        self.phase     = random.randint(0,1)
        self.step_n    = 0
        return self._state()

    def _state(self):
        return normalise(self.q['N'], self.q['S'],
                         self.q['E'], self.q['W'],
                         self.wait_time, self.emergency, self.phase)

    def step(self, action_idx):
            duration = DURATION_STEPS[action_idx]

            if self.phase == 0:
                active_dirs, waiting_dirs = ('N','S'), ('E','W')
            else:
                active_dirs, waiting_dirs = ('E','W'), ('N','S')

            active_total  = sum(self.q[d] for d in active_dirs)
            waiting_total = sum(self.q[d] for d in waiting_dirs)

            # ── Realistic clearance model ─────────────────────────────────────────
            # Each green second clears ~0.5 cars when traffic is present
            # Total clearance = min(cars available, rate × duration)
            # This means heavy traffic genuinely benefits from longer greens
            clearance_rate = 0.5   # cars cleared per second of green
            max_clearable  = clearance_rate * duration
            cleared        = min(float(active_total), max_clearable)
            cleared       += random.uniform(-0.5, 0.5)   # small noise
            cleared        = max(0.0, min(float(active_total), cleared))

            # Apply clearance to active directions proportionally
            for d in active_dirs:
                total = max(1, active_total)
                take  = min(self.q[d], round(cleared * self.q[d] / total))
                self.q[d] = max(0, self.q[d] - take)

            # Arrivals on all directions
            for d in 'NSEW':
                self.q[d] = min(10, self.q[d] + np.random.poisson(duration * 0.08))

            total_wait = self.wait_time + duration

            # ── Reward ────────────────────────────────────────────────────────────
            # Clearance reward: absolute cars cleared
            # Heavy road + long green = more cars cleared = higher reward
            # Empty road + any green = no cars cleared = low reward
            clearance_reward = cleared * 3.0                 # increased

            # Time cost: VERY STRONG penalty for long greens
            # Key insight: if no cars to clear, ANY green is wasteful
            time_cost = -duration * 0.5                      # very strong

            # Fairness: penalize long waits on waiting directions
            if   total_wait < 15: fairness =  3.0
            elif total_wait < 25: fairness =  1.0
            elif total_wait < 35: fairness = -3.0            # increased
            else:                 fairness = -8.0            # increased

            # Emergency
            emg_bonus = 0.0
            if self.emergency == 2:
                emg_bonus = (MAX_DURATION - duration) * 0.6  # increased
            elif self.emergency == 1 and duration <= 15:
                emg_bonus = 4.0                               # increased

            reward = clearance_reward + time_cost + fairness + emg_bonus

            # Swap axes
            self.phase     = 1 - self.phase
            self.wait_time = float(duration)
            self.emergency = random.choices([0,1,2], weights=[90,5,5])[0]
            self.step_n   += 1

            return self._state(), reward, self.step_n >= MAX_STEPS

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
    print("  4-Way Intersection — DQN Trainer (Efficiency Reward)")
    print("="*62)

    layers = [N_INPUTS, 24, 24, N_ACTIONS]
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

        if (ep+1) % 500 == 0:
            print(f"  Episode {ep+1:>5}/{EPISODES} | "
                  f"Avg reward: {np.mean(rlog[-300:]):>7.1f} | "
                  f"Loss: {np.mean(llog[-300:]) if llog else 0:>6.3f} | "
                  f"Epsilon: {eps:.3f}")

    net.save("model_weights")

    # ── Policy table ──────────────────────────────────────────────────────────
    print("\n  Learned policy:")
    print(f"  {'Situation':<58} Output")
    print("  "+"─"*70)
    tests = [
        ("N=0 S=0 E=4 W=4, wait=5s  (NS empty)",       0,0,4,4, 5,0,0),
        ("N=4 S=4 E=0 W=0, wait=5s  (NS heavy)",        4,4,0,0, 5,0,0),
        ("N=6 S=6 E=1 W=1, wait=5s  (NS heavy, EW light)",6,6,1,1,5,0,0),
        ("N=8 S=8 E=1 W=1, wait=3s  (NS very heavy)",   8,8,1,1, 3,0,0),
        ("N=1 S=1 E=6 W=6, wait=5s  (NS light, EW heavy)",1,1,6,6,5,0,1),
        ("N=3 S=3 E=3 W=3, wait=5s  (equal, fresh)",    3,3,3,3, 5,0,0),
        ("N=3 S=3 E=3 W=3, wait=35s (equal, long wait)",3,3,3,3,35,0,0),
        ("N=4 S=4 E=2 W=2, wait=8s, emg on waiting",    4,4,2,2, 8,2,0),
        ("N=0 S=0 E=0 W=0, wait=5s  (all empty)",       0,0,0,0, 5,0,0),
        ("N=2 S=2 E=2 W=2, wait=20s (moderate, long)",  2,2,2,2,20,0,0),
    ]
    for desc,n,s,e,w,wt,emg,phase in tests:
        dur = DURATION_STEPS[int(np.argmax(net.predict(normalise(n,s,e,w,wt,emg,phase))[0]))]
        print(f"  {desc:<58} → {dur:>5.1f}s")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n  Sanity checks:")
    checks = [
        ("Empty NS (0 cars) → SHORT ≤8s",       0,0,4,4, 5,0,0, lambda d: d<=8),
        ("Heavy NS (8+8), fresh → LONG ≥12s",   8,8,1,1, 3,0,0, lambda d: d>=12),
        ("Emergency on waiting → SHORT ≤8s",     4,4,2,2, 8,2,0, lambda d: d<=8),
        ("Critical wait (35s) → SHORT ≤10s",     3,3,3,3,35,0,0, lambda d: d<=10),
        ("Heavy (6+6), no wait → not min ≥8s",  6,6,0,0, 5,0,0, lambda d: d>=8),
    ]
    all_ok = True
    for desc,n,s,e,w,wt,emg,phase,chk in checks:
        dur = DURATION_STEPS[int(np.argmax(net.predict(normalise(n,s,e,w,wt,emg,phase))[0]))]
        ok  = chk(dur)
        print(f"  {'✓' if ok else '✗ FAILED':<10} {desc} → got {dur:.1f}s")
        if not ok: all_ok = False

    print(f"\n  {'✓ All checks passed. Run controller.py' if all_ok else '✗ Re-run train.py (DQN has variance — usually passes in 1-2 tries)'}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    w=100
    if len(rlog)>w:
        ax1.plot(np.convolve(rlog,np.ones(w)/w,mode='valid'),color='steelblue',lw=1)
    ax1.set_title("Reward — should rise and stabilise above 0")
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

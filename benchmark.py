"""
Benchmark: AI Controller vs. Fixed-Timer Controllers
====================================================
Compares performance metrics:
  - AI Controller (trained model)
  - Fixed Timer: 20s cycles
  - Fixed Timer: 30s cycles
  - Fixed Timer: 40s cycles

Metrics measured:
  - Average wait time (seconds)
  - Maximum wait time (seconds)
  - Total vehicles cleared
  - Average queue length
  - Throughput (vehicles per hour)

Run: python benchmark.py
Output: benchmark_results.png + console summary
"""

import numpy as np
import random
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# ── Import components from train.py ─────────────────────────────────────────

# We need to import from train.py without running the training
# Save original argv, replace with empty to prevent train.py from running
original_argv = sys.argv
sys.argv = ['train.py']  # Prevent if __name__ == '__main__' block

# Now import the components we need
from train import (NeuralNetwork, decode_action, normalise, 
                   Simulator, N_INPUTS, N_ACTIONS, DURATIONS, MAX_WAIT,
                   N_DURATIONS)

# Restore argv
sys.argv = original_argv

# ── Fixed Timer Controller ───────────────────────────────────────────────────

class FixedTimerController:
    """Simple fixed-timer that alternates NS and EW with fixed duration."""
    def __init__(self, duration):
        self.duration = duration
        self.current_axis = 'NS'  # Start with NS
    
    def decide(self, ns_wait, ew_wait, n, s, e, w):
        # Alternate between NS and EW
        axis = self.current_axis
        self.current_axis = 'EW' if axis == 'NS' else 'NS'
        return axis, self.duration

# ── AI Controller (loaded from trained model) ──────────────────────────────

class AIController:
    def __init__(self, weights_path="model_weights.npz"):
        if not os.path.exists(weights_path):
            print(f"[Error] {weights_path} not found. Run train.py first.")
            sys.exit(1)
        self.network = NeuralNetwork([N_INPUTS, 32, 32, N_ACTIONS])
        self.network.load(weights_path)
        print(f"[AI] Model loaded from {weights_path}")
    
    def decide(self, ns_wait, ew_wait, n, s, e, w):
        state = normalise(n, s, e, w, ns_wait, ew_wait)
        q_vals = self.network.predict(state)[0]
        action = int(np.argmax(q_vals))
        axis, duration = decode_action(action)
        
        # Apply same overrides as controller.py
        ns_total = n + s
        ew_total = e + w
        
        # Override: empty axis check
        if axis == 'NS' and ns_total == 0 and ew_total > 0:
            axis = 'EW'
            duration = min(25.0, max(5.0, 5.0 + ew_total * 2.0))
        elif axis == 'EW' and ew_total == 0 and ns_total > 0:
            axis = 'NS'
            duration = min(25.0, max(5.0, 5.0 + ns_total * 2.0))
        
        # Override: MAX_WAIT exceeded
        if ew_wait >= MAX_WAIT and axis != 'EW':
            axis = 'EW'
            duration = DURATIONS[len(DURATIONS) // 2]
        elif ns_wait >= MAX_WAIT and axis != 'NS':
            axis = 'NS'
            duration = DURATIONS[len(DURATIONS) // 2]
        
        return axis, duration

# ── Benchmark Simulation ────────────────────────────────────────────────────

class BenchmarkSimulator(Simulator):
    """Extended simulator that tracks detailed metrics."""
    def __init__(self):
        super().__init__()
        self.metrics = {
            'wait_times': [],      # List of wait times when vehicles are cleared
            'queue_lengths': [],  # Queue lengths at each step
            'clearances': 0,      # Total vehicles cleared
            'max_wait': 0,        # Maximum wait time observed
        }
    
    def reset_metrics(self):
        self.metrics = {
            'wait_times': [],
            'queue_lengths': [],
            'clearances': 0,
            'max_wait': 0,
        }
    
    def step_with_controller(self, controller):
        """Run one step using the provided controller."""
        # Controller makes decision
        axis, duration = controller.decide(
            self.ns_wait, self.ew_wait,
            self.q['N'], self.q['S'], self.q['E'], self.q['W']
        )
        
        # Record wait times BEFORE clearing (these are the actual wait times)
        if axis == 'NS':
            for d in ['N', 'S']:
                if self.q[d] > 0:
                    self.metrics['wait_times'].append(self.ns_wait)
        else:
            for d in ['E', 'W']:
                if self.q[d] > 0:
                    self.metrics['wait_times'].append(self.ew_wait)
        
        # Track max wait
        self.metrics['max_wait'] = max(self.metrics['max_wait'], 
                                        self.ns_wait, self.ew_wait)
        
        # Use parent step logic (but we need to extract clearance)
        ns_total = self.q['N'] + self.q['S']
        ew_total = self.q['E'] + self.q['W']
        
        if axis == 'NS':
            active_total = ns_total
            active_dirs = ('N', 'S')
        else:
            active_total = ew_total
            active_dirs = ('E', 'W')
        
        # Clearance
        clearance_rate = 0.5
        cleared = min(float(active_total), clearance_rate * duration + random.uniform(-0.3, 0.3))
        cleared = max(0.0, cleared)
        
        for d in active_dirs:
            total = max(1, active_total)
            take = min(self.q[d], round(cleared * self.q[d] / total))
            self.q[d] = max(0, self.q[d] - take)
        
        self.metrics['clearances'] += cleared
        
        # Arrivals
        for d in 'NSEW':
            self.q[d] = min(10, self.q[d] + np.random.poisson(duration * 0.08))
        
        # Update wait times
        if axis == 'NS':
            self.ns_wait = 0.0
            self.ew_wait = min(MAX_WAIT * 1.5, self.ew_wait + duration)
        else:
            self.ew_wait = 0.0
            self.ns_wait = min(MAX_WAIT * 1.5, self.ns_wait + duration)
        
        # Track queue lengths
        total_queue = sum(self.q.values())
        self.metrics['queue_lengths'].append(total_queue)
        
        # New emergency for next step
        self.emergency = random.choices([0,1,2], weights=[90,5,5])[0]
        
        return axis, duration

def run_benchmark(controller, num_episodes=100, steps_per_episode=60):
    """Run benchmark for a given controller."""
    all_wait_times = []
    all_max_waits = []
    all_clearances = []
    all_avg_queues = []
    
    for episode in range(num_episodes):
        sim = BenchmarkSimulator()
        sim.reset()
        sim.reset_metrics()
        
        for step in range(steps_per_episode):
            sim.step_with_controller(controller)
        
        # Collect metrics
        if sim.metrics['wait_times']:
            all_wait_times.append(np.mean(sim.metrics['wait_times']))
        else:
            all_wait_times.append(0)
        
        all_max_waits.append(sim.metrics['max_wait'])
        all_clearances.append(sim.metrics['clearances'])
        all_avg_queues.append(np.mean(sim.metrics['queue_lengths']))
    
    return {
        'avg_wait_time': np.mean(all_wait_times),
        'std_wait_time': np.std(all_wait_times),
        'avg_max_wait': np.mean(all_max_waits),
        'std_max_wait': np.std(all_max_waits),
        'avg_clearances': np.mean(all_clearances),
        'avg_queue_length': np.mean(all_avg_queues),
        'throughput_per_hour': np.mean(all_clearances) * (3600 / (steps_per_episode * 15)),  # Rough estimate
    }

# ── Main Benchmark ──────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("  Traffic Light Controller Benchmark")
    print("="*70)
    
    results = {}
    
    # ── AI Controller ─────────────────────────────────────────────────────
    print("\n[1/4] Running AI Controller...")
    try:
        ai_controller = AIController("model_weights.npz")
        ai_results = run_benchmark(ai_controller, num_episodes=100, steps_per_episode=60)
        results['AI Controller'] = ai_results
        print(f"      Done. Avg wait: {ai_results['avg_wait_time']:.1f}s")
    except Exception as e:
        print(f"      Error: {e}")
        print("      Skipping AI controller (run train.py first)")
    
    # ── Fixed Timer: 20s ─────────────────────────────────────────────────
    print("\n[2/4] Running Fixed Timer (20s)...")
    fixed_20 = FixedTimerController(duration=20)
    results['Fixed 20s'] = run_benchmark(fixed_20, num_episodes=100, steps_per_episode=60)
    print(f"      Done. Avg wait: {results['Fixed 20s']['avg_wait_time']:.1f}s")
    
    # ── Fixed Timer: 30s ─────────────────────────────────────────────────
    print("\n[3/4] Running Fixed Timer (30s)...")
    fixed_30 = FixedTimerController(duration=30)
    results['Fixed 30s'] = run_benchmark(fixed_30, num_episodes=100, steps_per_episode=60)
    print(f"      Done. Avg wait: {results['Fixed 30s']['avg_wait_time']:.1f}s")
    
    # ── Fixed Timer: 40s ─────────────────────────────────────────────────
    print("\n[4/4] Running Fixed Timer (40s)...")
    fixed_40 = FixedTimerController(duration=40)
    results['Fixed 40s'] = run_benchmark(fixed_40, num_episodes=100, steps_per_episode=60)
    print(f"      Done. Avg wait: {results['Fixed 40s']['avg_wait_time']:.1f}s")
    
    # ── Print Results ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  BENCHMARK RESULTS (100 episodes, 60 steps each)")
    print("="*70)
    print(f"\n{'Controller':<20} {'Avg Wait':>10} {'Max Wait':>10} {'Avg Queue':>10} {'Throughput':>12}")
    print("-"*70)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['avg_wait_time']:>8.1f}s {metrics['avg_max_wait']:>8.1f}s "
              f"{metrics['avg_queue_length']:>8.1f} {metrics['throughput_per_hour']:>10.1f}/hr")
    
    # ── Calculate Improvement ─────────────────────────────────────────────
    if 'AI Controller' in results:
        print("\n" + "="*70)
        print("  AI IMPROVEMENT vs. FIXED TIMERS")
        print("="*70)
        for name in ['Fixed 20s', 'Fixed 30s', 'Fixed 40s']:
            if name in results:
                wait_improvement = ((results[name]['avg_wait_time'] - results['AI Controller']['avg_wait_time']) 
                                    / results[name]['avg_wait_time'] * 100)
                print(f"  vs. {name:<10}: {wait_improvement:>+5.1f}% reduction in wait time")
    
    # ── Plot Results ─────────────────────────────────────────────────────
    plot_results(results)
    
    print("\n  Results saved to: benchmark_results.png")
    print("="*70)

def plot_results(results):
    """Create visualization of benchmark results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Traffic Controller Benchmark Results', fontsize=14, fontweight='bold')
    
    controllers = list(results.keys())
    x = np.arange(len(controllers))
    width = 0.35
    
    # Plot 1: Average Wait Time
    ax = axes[0, 0]
    wait_times = [results[c]['avg_wait_time'] for c in controllers]
    ax.bar(x, wait_times, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(controllers)])
    ax.set_ylabel('Seconds', fontweight='bold')
    ax.set_title('Average Wait Time', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(controllers, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(wait_times):
        ax.text(i, v + 0.5, f'{v:.1f}s', ha='center', fontweight='bold')
    
    # Plot 2: Maximum Wait Time
    ax = axes[0, 1]
    max_waits = [results[c]['avg_max_wait'] for c in controllers]
    ax.bar(x, max_waits, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(controllers)])
    ax.set_ylabel('Seconds', fontweight='bold')
    ax.set_title('Average Maximum Wait Time', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(controllers, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=40, color='r', linestyle='--', linewidth=2, label='MAX_WAIT=40s')
    ax.legend()
    for i, v in enumerate(max_waits):
        ax.text(i, v + 0.5, f'{v:.1f}s', ha='center', fontweight='bold')
    
    # Plot 3: Average Queue Length
    ax = axes[1, 0]
    queue_lengths = [results[c]['avg_queue_length'] for c in controllers]
    ax.bar(x, queue_lengths, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(controllers)])
    ax.set_ylabel('Vehicles', fontweight='bold')
    ax.set_title('Average Queue Length', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(controllers, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(queue_lengths):
        ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Plot 4: Throughput
    ax = axes[1, 1]
    throughputs = [results[c]['throughput_per_hour'] for c in controllers]
    ax.bar(x, throughputs, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(controllers)])
    ax.set_ylabel('Vehicles/Hour', fontweight='bold')
    ax.set_title('Estimated Throughput', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(controllers, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(throughputs):
        ax.text(i, v + 1, f'{v:.0f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()

'''Nexus Simulation — Monte Carlo scenarios, physics models.'''
import math, random, time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class Particle:
    x: float; y: float; z: float
    vx: float = 0; vy: float = 0; vz: float = 0
    mass: float = 1.0

class PhysicsEngine:
    def __init__(self, dt: float = 0.01, gravity: float = 9.81, drag: float = 0.1):
        self.dt = dt; self.gravity = gravity; self.drag = drag
    def step(self, particles: List[Particle]) -> List[Particle]:
        for p in particles:
            p.vz -= self.gravity * self.dt
            speed = math.sqrt(p.vx**2+p.vy**2+p.vz**2)
            if speed > 0:
                drag = self.drag * speed
                p.vx -= drag * p.vx/speed * self.dt
                p.vy -= drag * p.vy/speed * self.dt
                p.vz -= drag * p.vz/speed * self.dt
            p.x += p.vx * self.dt; p.y += p.vy * self.dt; p.z += p.vz * self.dt
        return particles
    def kinetic_energy(self, p: Particle) -> float:
        return 0.5 * p.mass * (p.vx**2 + p.vy**2 + p.vz**2)

class MonteCarlo:
    def __init__(self, seed: int = 42):
        random.seed(seed); self.results = []
    def run(self, simulate_fn, trials: int = 1000) -> Dict:
        successes = 0; values = []
        for _ in range(trials):
            result = simulate_fn()
            values.append(result)
            if result.get('success', False): successes += 1
        values_sorted = sorted(values, key=lambda x: x.get('value', 0))
        return {
            'success_rate': successes/trials, 'mean': sum(v.get('value',0) for v in values)/trials,
            'p50': values_sorted[trials//2]['value'] if values_sorted else 0,
            'p10': values_sorted[max(0,trials//10)]['value'] if len(values_sorted)>10 else 0,
            'p90': values_sorted[min(trials-1,9*trials//10)]['value'] if values_sorted else 0,
        }

class EnvironmentModel:
    def __init__(self, current_mps: float = 0.5, visibility_m: float = 10.0):
        self.current = current_mps; self.visibility = visibility_m
    def sensor_reading(self, true_value: float, sensor_noise: float = 0.1) -> float:
        return true_value + random.gauss(0, sensor_noise) + random.gauss(0, self.current * 0.05)
    def detect_range(self, target_strength: float, noise_floor: float = 0.01) -> float:
        return target_strength / noise_floor if noise_floor > 0 else float('inf')

def demo():
    print("=== Simulation ===")
    engine = PhysicsEngine(dt=0.1)
    particles = [Particle(i, 0, 10, 0, 5) for i in range(5)]
    for step in range(5):
        engine.step(particles)
    for p in particles[:3]:
        print(f"  Particle at ({p.x:.1f},{p.y:.1f},{p.z:.1f}) KE={engine.kinetic_energy(p):.1f}")
    mc = MonteCarlo()
    result = mc.run(lambda: {'success': random.random() > 0.3, 'value': random.gauss(100, 20)}, 1000)
    print(f"\nMonte Carlo: success={result['success_rate']:.1%}, mean={result['mean']:.1f}, p50={result['p50']:.1f}")
    env = EnvironmentModel(0.8, 8.0)
    for _ in range(3):
        print(f"  Sensor: {env.sensor_reading(5.0):.2f}")

if __name__ == "__main__": demo()

"""
particle.py

One particle (candidate relay-setting vector) for PSO.

Corrections vs. earlier version
-------------------------------
* Bounds are imported from constraints.py (single source of truth). The old
  code hard-coded TDS_MIN = 0.04 here vs 0.05 in constraints.py; particles
  clamped to 0.04 were then flagged infeasible forever. Now they agree.
* Optional constructive warm-start (set_constructive) so the swarm can be
  seeded near a feasible, coordinated point -- a standard, legitimate way to
  help a metaheuristic on a tightly-constrained coordination problem.
"""

import random

from constraints import (
    NUM_RELAYS, TDS_MIN, TDS_MAX, PICKUP_MIN, PICKUP_MAX,
)


class Particle:
    def __init__(self):
        self.TDS_MIN, self.TDS_MAX = TDS_MIN, TDS_MAX
        self.PICKUP_MIN, self.PICKUP_MAX = PICKUP_MIN, PICKUP_MAX

        self.position = []
        self.velocity = []
        for _ in range(NUM_RELAYS):
            self.position.extend([
                random.uniform(self.TDS_MIN, self.TDS_MAX),
                random.uniform(self.PICKUP_MIN, self.PICKUP_MAX),
            ])
            self.velocity.extend([
                random.uniform(-0.1, 0.1),
                random.uniform(-0.2, 0.2),
            ])

        self.best_position = self.position.copy()
        self.best_fitness = float("inf")

    def set_position(self, position):
        """Warm-start this particle at a given setting vector (clamped)."""
        self.position = list(position)
        self._clamp()
        self.best_position = self.position.copy()

    def update_velocity(self, global_best_position, inertia=0.7, c1=1.5, c2=1.5):
        for i in range(len(self.position)):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best_position[i] - self.position[i])
            v = inertia * self.velocity[i] + cognitive + social
            # velocity clamp (20% of each dimension's range) for stability
            lo = self.TDS_MIN if i % 2 == 0 else self.PICKUP_MIN
            hi = self.TDS_MAX if i % 2 == 0 else self.PICKUP_MAX
            vmax = 0.2 * (hi - lo)
            self.velocity[i] = max(-vmax, min(v, vmax))

    def update_position(self):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
        self._clamp()

    def _clamp(self):
        for relay_idx in range(0, len(self.position), 2):
            self.position[relay_idx] = max(self.TDS_MIN, min(self.position[relay_idx], self.TDS_MAX))
            self.position[relay_idx + 1] = max(self.PICKUP_MIN, min(self.position[relay_idx + 1], self.PICKUP_MAX))

    def print_particle(self):
        print("\nParticle\n--------------------------")
        for relay_idx in range(NUM_RELAYS):
            tds = self.position[2 * relay_idx]
            pickup = self.position[2 * relay_idx + 1]
            print(f"Relay {relay_idx + 1}: TDS={tds:.4f}, Pickup={pickup:.4f}")
        print(f"Best Fitness     : {self.best_fitness}\n--------------------------")
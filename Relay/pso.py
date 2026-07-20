"""
pso.py

Particle Swarm Optimization for coordinated relay-setting optimization.

Corrections vs. earlier version
-------------------------------
* optimize() now accepts and forwards `confidence`, so the confidence trust
  gate in objective.select_scenario actually fires. Without this the fallback
  group was unreachable (confidence defaulted to 1.0 every call).
* The extra penalty() call now receives the per-relay ZONE CURRENTS and is
  graded (it enforces the pairwise CTI margin). The objective already
  includes graded penalties, so this is consistent, not a conflicting flat
  1e6 cliff.
* One particle is warm-started with a constructive, near-coordinated guess so
  the swarm reliably reaches the tight feasible region.
"""

from particle import Particle
from objective import adaptive_objective, zone_currents_for_scenario, select_scenario
from constraints import penalty, NUM_RELAYS, TDS_MIN, TDS_MAX, PICKUP_MIN, PICKUP_MAX, COORDINATION_TIME
from relay_model import Relay

_relay = Relay()


def _constructive_seed(zone_currents, loading):
    """A feasible-leaning starting vector: operating times increasing upstream,
    each ~CTI apart, with pickups above the zone load current."""
    pos = []
    for k in range(NUM_RELAYS):
        pickup = max(PICKUP_MIN + 0.1, 1.25 * loading * (0.85 ** k) * 1.2)
        target = 0.25 + (COORDINATION_TIME + 0.1) * (NUM_RELAYS - 1 - k)
        ratio = zone_currents[k] / pickup
        tds = target * (ratio ** 0.02 - 1) / 0.14 if ratio > 1 else 0.5
        tds = min(max(tds, TDS_MIN), TDS_MAX)
        pos += [tds, pickup]
    return pos


class PSO:
    def __init__(self, num_particles=30, iterations=100):
        self.num_particles = num_particles
        self.iterations = iterations
        self.swarm = [Particle() for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_fitness = float("inf")

    def optimize(self, fault_current, loading, fault_type, fault_bus,
                 fault_impedance, pre_fault_voltage, fault_voltage,
                 confidence=1.0, verbose=False):

        zone_currents = zone_currents_for_scenario(fault_type, confidence)
        # warm-start the first particle near a coordinated solution
        self.swarm[0].set_position(_constructive_seed(zone_currents, loading))

        for iteration in range(self.iterations):
            for particle in self.swarm:
                fitness = adaptive_objective(
                    particle.position, fault_current, loading, fault_type,
                    fault_bus, fault_impedance, pre_fault_voltage, fault_voltage,
                    confidence,
                )
                fitness += penalty(particle.position, zone_currents)

                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

            for particle in self.swarm:
                particle.update_velocity(self.global_best_position)
                particle.update_position()

            if verbose:
                print(f"Iteration {iteration+1:03d} | Best = {self.global_best_fitness:.5f}")

        return self.global_best_position, self.global_best_fitness
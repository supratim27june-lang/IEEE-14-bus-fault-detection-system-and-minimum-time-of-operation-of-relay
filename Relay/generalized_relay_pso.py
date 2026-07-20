import argparse
import math
import random
from typing import List, Optional, Tuple

from relay_model import Relay


class Particle:
    def __init__(self, bounds: List[Tuple[float, float]], rng: random.Random):
        self.position = [rng.uniform(low, high) for low, high in bounds]
        self.velocity = [
            rng.uniform(-(high - low) * 0.1, (high - low) * 0.1)
            for low, high in bounds
        ]
        self.best_position = self.position.copy()
        self.best_fitness = float("inf")


class GeneralizedRelayPSO:
    def __init__(
        self,
        fault_current: float,
        bounds: List[Tuple[float, float]],
        num_particles: int = 30,
        iterations: int = 20,
        seed: int = 42,
        inertia_weight: float = 0.7,
        cognitive_weight: float = 1.5,
        social_weight: float = 1.5,
        loading: Optional[float] = None,
        fault_type: Optional[str] = None,
        fault_bus: Optional[int] = None,
        fault_impedance: Optional[float] = None,
        pre_fault_voltage: Optional[float] = None,
        fault_voltage: Optional[float] = None,
    ):
        self.fault_current = fault_current
        self.bounds = bounds
        self.num_particles = num_particles
        self.iterations = iterations
        self.rng = random.Random(seed)
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.loading = loading
        self.fault_type = fault_type
        self.fault_bus = fault_bus
        self.fault_impedance = fault_impedance
        self.pre_fault_voltage = pre_fault_voltage
        self.fault_voltage = fault_voltage
        self.relay = Relay()

        self.swarm = [Particle(bounds, self.rng) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_fitness = float("inf")

    def _clip_position(self, position: List[float]) -> List[float]:
        clipped = []
        for value, (low, high) in zip(position, self.bounds):
            clipped.append(max(low, min(high, value)))
        return clipped

    def _fitness(self, position: List[float]) -> float:
        pickup_current, tds = position

        if pickup_current <= 0 or tds <= 0:
            return float("inf")

        operating_time = self.relay.relay_operating_time(self.fault_current, pickup_current, tds)

        if not math.isfinite(operating_time):
            return float("inf")

        if operating_time >= 9999:
            return 9999.0 + abs(self.fault_current - pickup_current) * 10.0

        penalty = 0.0

        if self.loading is not None:
            if self.loading > 0.90:
                if pickup_current < 2.5:
                    penalty += 2.0
            elif self.loading > 0.75:
                if pickup_current < 2.0:
                    penalty += 1.0

        if self.fault_type in ["LLL", "LLLG"]:
            if operating_time > 0.25:
                penalty += 3.0
        elif self.fault_type == "SLG":
            if operating_time > 0.40:
                penalty += 2.0

        if self.fault_bus is not None and self.fault_bus >= 10:
            if tds > 0.5:
                penalty += 1.0

        return operating_time + penalty

    def _raw_operating_time(self, position: List[float]) -> float:
        pickup_current, tds = position
        return self.relay.relay_operating_time(self.fault_current, pickup_current, tds)

    def _update_particle(self, particle: Particle) -> None:
        r1 = self.rng.random()
        r2 = self.rng.random()

        for index in range(len(particle.position)):
            low, high = self.bounds[index]
            max_velocity = (high - low) * 0.2

            inertia = self.inertia_weight * particle.velocity[index]
            cognitive = self.cognitive_weight * r1 * (particle.best_position[index] - particle.position[index])
            social = self.social_weight * r2 * (self.global_best_position[index] - particle.position[index])

            particle.velocity[index] = inertia + cognitive + social
            particle.velocity[index] = max(-max_velocity, min(max_velocity, particle.velocity[index]))

            particle.position[index] += particle.velocity[index]
            particle.position[index] = max(low, min(high, particle.position[index]))

    def optimize(self) -> Tuple[List[float], float]:
        print("\nStarting generalized PSO optimization for the standard relay...\n")

        for particle in self.swarm:
            fitness = self._fitness(particle.position)
            particle.best_fitness = fitness
            particle.best_position = particle.position.copy()

            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()

        for iteration in range(self.iterations):
            for particle in self.swarm:
                self._update_particle(particle)

                fitness = self._fitness(particle.position)

                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

            print(
                f"Iteration {iteration + 1:02d} | Best Time = {self.global_best_fitness:.5f} sec"
            )

        return self.global_best_position, self.global_best_fitness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generalized PSO optimizer for a standard relay")
    parser.add_argument("--fault-current", type=float, required=True, help="Fault current in kA")
    parser.add_argument("--particles", type=int, default=25, help="Number of particles in the swarm")
    parser.add_argument("--iterations", type=int, default=40, help="Number of PSO iterations")
    parser.add_argument("--pickup-min", type=float, default=None, help="Lower bound for pickup current")
    parser.add_argument("--pickup-max", type=float, default=None, help="Upper bound for pickup current")
    parser.add_argument("--tds-min", type=float, default=0.05, help="Lower bound for TDS")
    parser.add_argument("--tds-max", type=float, default=1.2, help="Upper bound for TDS")
    parser.add_argument("--loading", type=float, default=None, help="Loading in pu for penalty handling")
    parser.add_argument("--fault-type", type=str, default=None, help="Predicted fault type for penalty handling")
    parser.add_argument("--fault-bus", type=int, default=None, help="Fault bus index for penalty handling")
    parser.add_argument("--fault-impedance", type=float, default=None, help="Fault impedance for compatibility")
    parser.add_argument("--pre-fault-voltage", type=float, default=None, help="Pre-fault voltage for compatibility")
    parser.add_argument("--fault-voltage", type=float, default=None, help="Fault voltage for compatibility")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fault_current = args.fault_current
    pickup_min = args.pickup_min if args.pickup_min is not None else max(0.1, fault_current * 0.1)
    pickup_max = args.pickup_max if args.pickup_max is not None else fault_current * 0.95

    if pickup_max <= pickup_min:
        raise ValueError("pickup_max must be greater than pickup_min")

    bounds = [(pickup_min, pickup_max), (args.tds_min, args.tds_max)]

    optimizer = GeneralizedRelayPSO(
        fault_current=fault_current,
        bounds=bounds,
        num_particles=args.particles,
        iterations=args.iterations,
        seed=args.seed,
        loading=args.loading,
        fault_type=args.fault_type,
        fault_bus=args.fault_bus,
        fault_impedance=args.fault_impedance,
        pre_fault_voltage=args.pre_fault_voltage,
        fault_voltage=args.fault_voltage,
    )

    best_position, best_time = optimizer.optimize()
    pickup_current, tds = best_position
    raw_operating_time = optimizer._raw_operating_time(best_position)

    print("\nOptimization complete")
    print(f"Best pickup current: {pickup_current:.4f}")
    print(f"Best TDS: {tds:.4f}")
    print(f"Estimated objective value: {best_time:.4f}")
    print(f"Operating time (best_time - penalty): {raw_operating_time:.4f}")


if __name__ == "__main__":
    main()

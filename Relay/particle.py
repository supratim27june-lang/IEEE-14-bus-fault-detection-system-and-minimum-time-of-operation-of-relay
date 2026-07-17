"""
particle.py

Particle class for Particle Swarm Optimization (PSO)
used to optimize relay settings.
"""

import random


class Particle:
    """
    Represents one particle (candidate solution) in the swarm.
    """

    def __init__(self):

        # -----------------------------
        # Search Space Limits
        # -----------------------------
        self.TDS_MIN = 0.05
        self.TDS_MAX = 1.20

        self.PICKUP_MIN = 0.50
        self.PICKUP_MAX = 5.00

        # -----------------------------
        # Random Initial Position
        # -----------------------------
        self.position = [

            random.uniform(self.TDS_MIN, self.TDS_MAX),

            random.uniform(self.PICKUP_MIN, self.PICKUP_MAX)

        ]

        # -----------------------------
        # Initial Velocity
        # -----------------------------
        self.velocity = [

            random.uniform(-0.1, 0.1),

            random.uniform(-0.2, 0.2)

        ]

        # -----------------------------
        # Personal Best
        # -----------------------------
        self.best_position = self.position.copy()

        self.best_fitness = float("inf")

    # -------------------------------------------------
    # Update Velocity
    # -------------------------------------------------

    def update_velocity(
        self,
        global_best_position,
        inertia=0.7,
        c1=2.0,
        c2=2.0
    ):

        for i in range(2):

            r1 = random.random()
            r2 = random.random()

            cognitive = (
                c1
                * r1
                * (self.best_position[i] - self.position[i])
            )

            social = (
                c2
                * r2
                * (global_best_position[i] - self.position[i])
            )

            self.velocity[i] = (

                inertia * self.velocity[i]

                + cognitive

                + social

            )

    # -------------------------------------------------
    # Update Position
    # -------------------------------------------------

    def update_position(self):

        for i in range(2):

            self.position[i] += self.velocity[i]

        # -----------------------------
        # Clamp TDS
        # -----------------------------
        self.position[0] = max(
            self.TDS_MIN,
            min(self.position[0], self.TDS_MAX)
        )

        # -----------------------------
        # Clamp Pickup
        # -----------------------------
        self.position[1] = max(
            self.PICKUP_MIN,
            min(self.position[1], self.PICKUP_MAX)
        )

    # -------------------------------------------------
    # Display Particle
    # -------------------------------------------------

    def print_particle(self):

        print("\nParticle")

        print("--------------------------")

        print(f"TDS              : {self.position[0]:.4f}")

        print(f"Pickup Current   : {self.position[1]:.4f}")

        print(f"Velocity         : {self.velocity}")

        print(f"Best Position    : {self.best_position}")

        print(f"Best Fitness     : {self.best_fitness}")

        print("--------------------------")
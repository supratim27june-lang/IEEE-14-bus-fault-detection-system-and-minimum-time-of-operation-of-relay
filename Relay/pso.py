"""
pso.py

Particle Swarm Optimization
for Relay Setting Optimization
"""

from particle import Particle
from objective import adaptive_objective
from constraints import penalty


class PSO:

    def __init__(self,
                 num_particles=30,
                 iterations=50):

        self.num_particles = num_particles
        self.iterations = iterations

        # Create swarm
        self.swarm = [Particle() for _ in range(num_particles)]

        # Global Best
        self.global_best_position = None
        self.global_best_fitness = float("inf")

    #########################################################

    def optimize(
        self,
        fault_current,
        loading,
        fault_type,
        fault_bus,
        fault_impedance,
        pre_fault_voltage,
        fault_voltage
    ):

        print("\nStarting PSO Optimization...\n")

        for iteration in range(self.iterations):

            for particle in self.swarm:

                # Calculate fitness
                fitness = adaptive_objective(
                    particle.position,
                    fault_current,
                    loading,
                    fault_type,
                    fault_bus,
                    fault_impedance,
                    pre_fault_voltage,
                    fault_voltage
                )

                # Add penalty if constraints violated
                fitness += penalty(
                    particle.position,
                    fault_current
                )

                #######################################

                # Personal Best

                if fitness < particle.best_fitness:

                    particle.best_fitness = fitness

                    particle.best_position = particle.position.copy()

                #######################################

                # Global Best

                if fitness < self.global_best_fitness:

                    self.global_best_fitness = fitness

                    self.global_best_position = particle.position.copy()

            ###########################################

            # Move particles

            for particle in self.swarm:

                particle.update_velocity(
                    self.global_best_position
                )

                particle.update_position()

            ###########################################

            print(
                f"Iteration {iteration+1:02d}"
                f" | Best Time = {self.global_best_fitness:.5f} sec"
            )

        ###############################################

        return (
            self.global_best_position,
            self.global_best_fitness
        )
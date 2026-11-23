"""Interaction force estimation for the exported UAV."""
import numpy as np

class ThrustResidual:
    """Smooth the force exerted on the environment, in world XY coordinates."""

    def __init__(self, mass):
        self.mass = mass
        self.samples = []

    def observe(self, rotation, specific_force, previous_thrust):
        thrust_body = np.array([0., 0., np.sum(previous_thrust)])
        residual = rotation @ (thrust_body - self.mass * specific_force)
        self.samples.append(residual[:2])

    def take(self):
        estimate = np.median(self.samples, axis=0) if self.samples else np.zeros(2)
        self.samples.clear()
        return estimate

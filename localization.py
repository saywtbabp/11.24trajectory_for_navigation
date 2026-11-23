"""IMU propagation and wall range registration."""
import numpy as np

G = np.array([0., 0., 9.81])
WALL_THICKNESS = 0.05

class Localization:
    """Known takeoff position, IMU attitude and scan-to-coarse-wall registration."""
    def __init__(self, arena, altitude):
        self.position = np.r_[arena.start, altitude]
        self.velocity = np.zeros(3)
        self.rotation = np.eye(3)
        self.scan_origin = self.position.copy()
        self.boxes = arena.boxes(WALL_THICKNESS)
        self.wall_segments = arena.walls if hasattr(arena, "bounds") else None
        self.hits = 0

    def predict(self, acc, gyro, dt, *, near_hover):
        rotation = self.rotation
        # Under commanded acceleration, the accelerometer also sees vehicle
        # motion. Using it as gravity tilts the estimate and biases dead reckoning.
        rate = gyro
        if near_hover and np.linalg.norm(self.velocity[:2]) < .18:
            magnitude = np.linalg.norm(acc)
            if abs(magnitude-9.81) < .35:
                gravity_error = np.cross(acc / magnitude, rotation[2])
                rate = gyro + .10 * gravity_error
        axis = rate * dt
        x, y, z = axis
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        self.rotation = rotation @ (np.eye(3) + skew + .5 * (skew @ skew))
        world_acc = self.rotation @ acc - G
        self.position += self.velocity * dt + .5 * world_acc * dt**2
        self.velocity += world_acc * dt

    def match_vertical_faces(self, predicted, headings, surface_normals):
        walls = self.wall_segments[np.abs(self.wall_segments[:, 0]-self.wall_segments[:, 2]) < 1e-8]
        centers = walls[:, 0]
        lo = np.minimum(walls[:, 1], walls[:, 3]) + .10
        hi = np.maximum(walls[:, 1], walls[:, 3]) - .10
        normal_return = (np.abs(headings) > .60) & (surface_normals[:, 0] > .96)
        predicted = predicted[normal_return]
        headings = headings[normal_return]
        faces = centers - np.sign(headings[:, None]) * WALL_THICKNESS
        errors = faces - predicted[:, 0, None]
        in_span = (predicted[:, 1, None] > lo) & (predicted[:, 1, None] < hi)
        residual = np.where(in_span, np.abs(errors), np.inf)
        best = np.argmin(residual, axis=1)
        indices = np.arange(len(best))
        matched = residual[indices, best] < .19
        return errors[indices[matched], best[matched]]

    def correct(self, rays, ranges, scan_dt):
        directions = rays @ self.rotation.T
        altitude = -directions[-1, 2] * ranges[-1] if ranges[-1] > 0 else self.position[2]
        horizontal_ranges = ranges[:-1]
        if self.wall_segments is not None:
            # When the UAV tilts, nominally horizontal rays can hit the floor.
            # Project returns into world height before matching wall faces.
            hit_height = altitude + directions[:-1, 2] * horizontal_ranges
            horizontal_ranges = np.where(hit_height > .18, horizontal_ranges, -1.)
        valid = horizontal_ranges > .05
        relative = directions[:-1][valid, :2] * horizontal_ranges[valid, None]
        # Register against the four outer wall faces. Use only returns with
        # near-normal incidence and a small residual, excluding the door.
        dx, dy = directions[:-1][valid, :2].T
        predicted = self.position[:2] + relative
        if self.wall_segments is not None:
            endpoints = directions[:-1, :2] * horizontal_ranges[:, None]
            forward = np.roll(endpoints, -2, axis=0) - endpoints
            backward = endpoints - np.roll(endpoints, 2, axis=0)
            lengths_f = np.linalg.norm(forward, axis=1)
            lengths_b = np.linalg.norm(backward, axis=1)
            span = forward + backward
            length = np.linalg.norm(span, axis=1)
            flat = ((lengths_f < .70) & (lengths_b < .70) &
                    (np.sum(forward*backward, axis=1) > .87*lengths_f*lengths_b))
            normals = np.abs(span[:, ::-1]) / np.maximum(length[:, None], 1e-6)
            normals[~flat] = 0.
            surface_normals = normals[valid]
        corrections = []
        self.hits = 0
        for dim, headings, faces in ((0, dx, (self.boxes[2, 2], self.boxes[3, 0])),
                                      (1, dy, (self.boxes[0, 3], self.boxes[1, 1]))):
            if self.wall_segments is not None and dim == 0:
                residuals = self.match_vertical_faces(predicted, headings, surface_normals)
            elif self.wall_segments is not None:
                residuals = []
                # A true external horizontal face produces a long cluster at
                # the same y. Endcaps on the partitions produce short runs.
                for sign, face in zip((-1, 1), faces):
                    mask = (headings * sign > .45) & (surface_normals[:, 1] > .98)
                    samples = predicted[mask]
                    if not len(samples):
                        continue
                    values = face - samples[:, 1]
                    near = np.abs(values) < 2.5
                    values, samples = values[near], samples[near]
                    if len(values) < 9:
                        continue
                    order = np.argsort(values)
                    sorted_values = values[order]
                    ends = np.searchsorted(sorted_values, sorted_values + .09)
                    groups = ends - np.arange(len(ends))
                    center = int(np.argmax(groups))
                    group = order[center:ends[center]]
                    # A nearby outer wall has a short sampled span because
                    # the scan only accepts rays near its normal.
                    near_face = abs(face-self.position[1]) < .85
                    minimum_span = .28 if near_face else 1.1
                    if len(group) >= 9 and np.ptp(samples[group, 0]) > minimum_span:
                        residuals.extend(values[group])
            else:
                residuals = []
                for sign, face in zip((-1, 1), faces):
                    error = face - predicted[:, dim]
                    matching = (headings * sign > .82) & (np.abs(error) < .20)
                    residuals.extend(error[matching])
            self.hits += len(residuals)
            corrections.append(np.median(residuals) if len(residuals) >= 6 else 0.)
        corrections.append(altitude - self.position[2])
        if self.wall_segments is not None:
            corrections = np.clip(corrections, -.12, .12)
        self.position += .72 * np.asarray(corrections)
        scan_velocity = (self.position - self.scan_origin) / scan_dt
        self.velocity = .76 * self.velocity + .24 * scan_velocity
        self.scan_origin = self.position.copy()
        return self.position[:2] + relative

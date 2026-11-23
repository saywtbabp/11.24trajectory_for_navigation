"""MuJoCo scene, sensor emulation, and the four-rotor interface."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from scene import WALL_HALF

MODEL = Path(__file__).with_name("tensegrity_real_model.xml")
ALTITUDE = 1.27
THICK = WALL_HALF
DOOR_WIDTH = 1.77
DT = 0.002
RECOVERY_INITIAL_FRICTION_NM = .07
RESISTANCE_RAMPS = {"effort_ramp": (15.5, RECOVERY_INITIAL_FRICTION_NM, 1.65),
                   "breakaway": (26.8, 2.15, .12)}
SCAN_ANGLES = np.linspace(-np.pi, np.pi, 720, endpoint=False)
RAYS = np.vstack((np.column_stack((np.cos(SCAN_ANGLES), np.sin(SCAN_ANGLES),
                                   np.zeros(len(SCAN_ANGLES)))), [0, 0, -1]))
ROTOR_POS = np.array([[0.08839, -0.08839], [0.08839, 0.08839],
                      [-0.08839, 0.08839], [-0.08839, -0.08839]])
MIXER = np.linalg.inv(np.vstack((np.ones(4), ROTOR_POS[:, 1], -ROTOR_POS[:, 0],
                                  [0.01302, -0.01302, 0.01302, -0.01302])))


class Simulator:
    def __init__(self, arena, fixtures, seed: int, noise: float):
        root = ET.parse(MODEL).getroot()
        sensors = root.find("sensor")
        for element in list(sensors):
            if element.tag.startswith("frame"):
                sensors.remove(element)
        actuators = root.find("actuator")
        for element in list(actuators):
            if element.tag != "general":
                actuators.remove(element)
        world = root.find("worldbody")
        for geom in world.iter("geom"):
            geom.set("group", "1")
        for body in world.findall("body"):
            xyz = np.fromstring(body.get("pos"), sep=" ")
            xyz[:2] += arena.start
            body.set("pos", " ".join(map(str, xyz)))

        left, bottom, right, top = getattr(arena, "bounds", (-8., -8., 8., 8.))
        ET.SubElement(world, "geom", name="floor", type="plane",
                      pos=f"{(left+right)/2} {(bottom+top)/2} 0",
                      size=f"{(right-left)/2+1} {(top-bottom)/2+1} .1", rgba=".24 .28 .31 1")
        for index, (x0, y0, x1, y1) in enumerate(arena.walls):
            vec = np.array([x1-x0, y1-y0])
            ET.SubElement(world, "geom", name=f"wall_{index}", type="box",
                          pos=f"{(x0+x1)/2} {(y0+y1)/2} 1.25",
                          euler=f"0 0 {np.degrees(np.arctan2(vec[1],vec[0]))}",
                          size=f"{np.linalg.norm(vec)/2} {THICK} 1.25",
                          rgba=".62 .67 .72 1")
        total_panel_mass = 0.
        snap_names = []
        rising_resistance_names = []
        for name, x, jamb, width, degrees, resistance, side in fixtures:
            hinge = ET.SubElement(world, "body", name=f"{name}_body",
                                  pos=f"{x} {jamb} 1.25")
            if resistance == "snap_open":
                snap_names.append(name)
                resistance = "ultralight"
            if resistance in RESISTANCE_RAMPS:
                rising_resistance_names.append((name, *RESISTANCE_RAMPS[resistance]))
                initial_friction = RESISTANCE_RAMPS[resistance][1]
                resistance = "resistance_profile"
            else:
                initial_friction = RECOVERY_INITIAL_FRICTION_NM
            friction = {"easy": .07, "binding": .07, "resistive": .95,
                        "stiff": 1.5, "very_stiff": 3.5, "ultralight": .001,
                        "heavy": .70, "light": .02,
                        "resistance_profile": initial_friction}[resistance]
            if resistance == "ultralight":
                mass, damping, armature = .015, .001, .0001
            elif resistance == "heavy":
                mass, damping, armature = 2.25, .40, .025
            elif resistance == "light":
                mass, damping, armature = .15, .07, .005
            else:
                mass, damping, armature = .75, .40, .025
            total_panel_mass += mass
            limit = 34 if resistance == "binding" else 90
            ET.SubElement(hinge, "joint", name=name, axis=f"0 0 {-side}", range=f"0 {limit}",
                          damping=str(damping), frictionloss=str(friction), armature=str(armature))
            # Keep the leaf clear of the box used to represent the jamb.
            ET.SubElement(hinge, "geom", name=f"{name}_panel", type="box",
                          size=f".025 {(width-.065)/2} 1.05", pos=f"0 {side*(width+.065)/2} 0",
                          mass=str(mass), friction=".35 .005 .0001", rgba=".77 .41 .15 1")
        self.model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mj.MjData(self.model)
        for name, _, _, _, degrees, _, _ in fixtures:
            self.data.joint(name).qpos[:] = np.radians(degrees)
        self.snap_joints = [self.data.joint(name) for name in snap_names]
        self.rising_resistance_joints = [
            (self.data.joint(name), self.model.jnt_dofadr[self.model.joint(name).id], start_deg, initial, final)
            for name, start_deg, initial, final in rising_resistance_names
        ]
        self.released_snaps = set()
        mj.mj_forward(self.model, self.data)
        self.mass = self.model.body_subtreemass[0] - total_panel_mass
        self.imu = self.data.site("imu")
        self.random = np.random.default_rng(seed)
        self.noise = noise
        self.hit_ids = np.empty(len(RAYS), np.int32)
        self.distances = np.empty(len(RAYS))
        self.sparse_panel_id = self.model.geom("D3_panel").id
        self.groups = np.array([1, 0, 0, 0, 0, 0], np.uint8)

    def inertial(self):
        acc = self.data.sensor("imu_acc").data.copy()
        gyro = self.data.sensor("imu_gyro").data.copy()
        return acc, gyro + self.random.normal(0, self.noise * 0.04, 3)

    def scan(self):
        # Only the instrument simulator uses the physical pose of the sensor.
        # The interface returns raw ranges. Hit IDs stay inside the ray API.
        directions = RAYS @ self.imu.xmat.reshape(3, 3).T
        mj.mj_multiRay(self.model, self.data, self.imu.xpos, directions[:-1].ravel(),
                       self.groups, 1, -1, self.hit_ids[:-1], self.distances[:-1], len(RAYS)-1, 17)
        # Use a separate range ray for altitude. The shared frustum culling
        # of the horizontal scan can discard the floor on larger maps.
        self.distances[-1] = mj.mj_ray(self.model, self.data, self.imu.xpos,
                                       directions[-1], self.groups, 1, -1, self.hit_ids[-1:])
        ranges = self.distances.copy()
        # D3 emulates intermittent close-range occlusion on the door only.
        # Hit identity is used inside the sensor model to remove returns.
        # Navigation receives ranges and never receives the identity.
        horizontal = ranges[:-1]
        door_hit = self.hit_ids[:-1] == self.sparse_panel_id
        if np.any(door_hit & (horizontal > 0) & (horizontal < .36)) and self.data.time % 1.4 < .82:
            horizontal[door_hit & (np.arange(len(horizontal)) % 90 != 0)] = -1.
        valid = ranges > 0
        ranges[valid] += self.random.normal(0, self.noise, valid.sum())
        return ranges

    def step(self, thrust):
        # Test fixture only. A preloaded hinge assists opening after the first
        # degree of travel, creating a fast swing after UAV contact.
        for index, joint in enumerate(self.snap_joints):
            angle = float(joint.qpos[0])
            if angle > np.radians(36):
                self.released_snaps.add(index)
            if index in self.released_snaps:
                joint.qfrc_applied[:] = 100. * max(0., np.pi/2-angle)
        # Test a tight sector of the hinge with no intermediate travel stop.
        for joint, dof, start_deg, initial, final in self.rising_resistance_joints:
            angle_deg = math.degrees(float(joint.qpos[0]))
            fraction = np.clip(angle_deg-start_deg, 0., 1.)
            self.model.dof_frictionloss[dof] = initial + fraction*(final-initial)
        self.data.ctrl[:] = thrust
        mj.mj_step(self.model, self.data)
        # Synchronize instrument sites after integration.
        mj.mj_kinematics(self.model, self.data)


def rotor_control(rotation, gyro, acc_world, mass):
    desired_z = np.r_[acc_world[:2], 9.81 + acc_world[2]]
    desired_z /= np.linalg.norm(desired_z)
    desired_y = np.cross(desired_z, [1., 0., 0.])
    desired_y /= np.linalg.norm(desired_y)
    target = np.column_stack((np.cross(desired_y, desired_z), desired_y, desired_z))
    error = target.T @ rotation - rotation.T @ target
    attitude_error = np.array([error[2, 1], error[0, 2], error[1, 0]]) * 0.5
    torque = -np.array([1.45, 1.45, 0.38]) * attitude_error
    torque -= np.array([0.16, 0.16, 0.09]) * gyro
    collective = mass * np.dot(np.r_[acc_world[:2], 9.81 + acc_world[2]], rotation[:, 2])
    return np.clip(MIXER @ np.r_[collective, torque], 0, 10)



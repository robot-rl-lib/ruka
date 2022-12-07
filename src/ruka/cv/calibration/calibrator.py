import cv2
import dataclasses
import numpy as np
import open3d as o3d
import time

from collections import defaultdict
from dataclasses import dataclass
from numpy.typing import NDArray
from ruka.robot.realsense import RealsenseCamera
from ruka.robot.xarm import XArmPosControlled, ControlMode
from ruka.util.x3d import compose_matrix_world
from scipy.optimize import least_squares
from tqdm.auto import tqdm
from typing import Tuple
from ruka.robot.perception import SensorSystem, SensorId

from .calibration import RGBDCameraCalibration, RobotCalibration
from .checkerboard import Checkerboard
from .loss import (
    EstimateBoardPositionsLoss,
    EstimateIntrinsicsLoss,
    EstimateGlobalBoardPositionLoss,
    EstimateColorParametersLoss,
    EstimateExtrinsicsLoss,
)
from .sensor import Sensor
from .transformation import Transformation


@dataclasses.dataclass
class CalibratorConfig:
    @dataclasses.dataclass
    class SLAMModelConfig:
        device: str
        block_resolution: int
        voxel_size: int
        block_count: int
        depth_scale: float
        depth_min: float
        depth_max: float
        odometry_distance_threshold: float
        trunc_voxel_multiplier: float

    n_frames_random_movement: int
    n_frames_static: int
    frames_interval_seconds: float
    static_frames_pos: np.array
    max_shifts: Tuple[float, float, float]
    max_angles: Tuple[float, float, float]
    home_pos: np.array
    checkerboard_size: Tuple[int, int]
    checkerboard_cell_size: float
    optimization_tolerance: float
    huber_loss_delta: float
    focal_length_init_value: float
    frame_shape: Tuple[int, int]

    slam_model: SLAMModelConfig


class Calibrator:
    def __init__(self, config: CalibratorConfig):
        self._config = config

    def collect_frames_random_move(
        self,
        robot_controller: XArmPosControlled,
        cameras: SensorSystem,
        gripper_camera_id: SensorId):
        """
        Collects frames for calibration using random robot movement

        Args:
            robot_controller: position controller to use
            camera: camera to capture frames from
            gripper_camera_id: id of camera on the gripper

        Returns:
            frames (List[dict]): list of collected frames
        """

        frames = []
        robot_controller.go_home()
        robot_controller.steady(ControlMode.POS)

        while len(frames) < self._config.n_frames_random_movement:
            print('Collected {}/{} frames'.format(len(frames), self._config.n_frames_random_movement))

            try:
                target = self._get_random_robot_position()
                robot_controller.set_pos(target[:3], target[3:])
                while not robot_controller.is_target_reached():
                    frame = self._collect_frame(robot_controller, cameras, [gripper_camera_id])
                    frames.append(frame)
            except Exception as e:
                print('Exception while collecting frames:', e)

                robot_controller.go_home()
                robot_controller.steady(ControlMode.POS)

        return frames

    def collect_frames_static(
        self,
        robot_controller: XArmPosControlled,
        cameras: SensorSystem):
        """
        Collects frames for calibration in static position

        Args:
            robot_controller: position controller to use
            camera: camera to capture frames from
            gripper_camera_id: id of camera on the gripper

        Returns:
            frames (List[dict]): list of collected frames
        """

        robot_controller.go_home()
        robot_controller.steady(ControlMode.POS)

        target = self._config.static_frames_pos
        robot_controller.set_pos(target[:3], target[3:])
        while not robot_controller.is_target_reached():
            time.sleep(self._config.frames_interval_seconds)

        frames = []
        for _ in tqdm(range(self._config.n_frames_static)):
            frames.append(self._collect_frame(robot_controller, cameras))

        return frames

    def _collect_frame(self, robot, cameras, filter_sensors=None):
        frame_full = cameras.capture()
        if filter_sensors is not None:
            frame = dict()
            for k in frame_full.keys():
                if k in filter_sensors:
                    frame[k] = frame_full[k]
        else:
            frame = frame_full

        frame['robot_pos'] = np.array(robot.pos + robot.angles)
        time.sleep(self._config.frames_interval_seconds)
        return frame

    def calibrate(
        self,
        frames_random_move,
        frames_static,
        gripper_camera_id) -> RobotCalibration:
        """
        Calibrates intrinsic and extrinsic camera parameters

        Args:
            frames_random_move: frames collected with 'collect_frames_random_move' method
            frames_static: frames collected with 'collect_frames_static' method
            gripper_camera_id: id of camera on the gripper

        Returns:
            calibration (RobotCalibration): all calibrated parameters
        """

        print('Calibrating gripper camera...')
        frame_data = self._extract_data_for_calibration(frames_random_move)
        gripper_camera_calib = self._calibrate_rgbd_camera(frame_data[gripper_camera_id])
        gripper_camera_to_tcp = self._calibrate_camera_to_tcp(frame_data[gripper_camera_id])

        print('Calibrating static cameras...')
        frame_data = self._extract_data_for_calibration(frames_static)

        board_positions = self._estimate_boards_positions(
            Sensor.create_from_intrinsics(gripper_camera_calib.depth_intrinsics),
            frame_data[gripper_camera_id])
        for f, board_pos in zip(frame_data[gripper_camera_id], board_positions):
            f.board_pos = board_pos

        camera_id_to_calibration = dict()
        for id, frames in frame_data.items():
            if id == gripper_camera_id:
                continue
            camera_id_to_calibration[id] = self._calibrate_rgbd_camera(frames)

        extrinsics = self._calibrate_extrinsics(frame_data)
        extrinsics = self._transform_extrinsics(
            extrinsics,
            frame_data,
            gripper_camera_to_tcp,
            gripper_camera_id)

        static_cameras = dict()
        for id, calib in camera_id_to_calibration.items():
            static_cameras[id] = StaticCamera(
                camera=calib,
                extrinsics_to_base=extrinsics[id],
            )

        return RobotCalibration(
            gripper_camera=gripper_camera_calib,
            gripper_camera_to_tcp=gripper_camera_to_tcp,
            static_cameras=static_cameras,
        )

    def _calibrate_rgbd_camera(self, frame_data):
        cb = self._make_checkerboard()

        sensor_init = np.array([
            self._config.focal_length_init_value,
            self._config.focal_length_init_value,
            self._config.frame_shape[1] // 2,
            self._config.frame_shape[0] // 2,
        ])

        depth_sensor = self._calibrate_depth_parameters(
            cb,
            sensor_init,
            frame_data)

        color_sensor, depth_to_color = self._calibrate_color_parameters(
            cb,
            sensor_init,
            frame_data)

        result = RGBDCameraCalibration(
            depth_intrinsics=depth_sensor.intrinsics_as_matrix(),
            color_intrinsics=color_sensor.intrinsics_as_matrix(),
            depth_to_color=depth_to_color.as_matrix(),
        )

        return result

    def _calibrate_extrinsics(self, frame_data):
        cb = self._make_checkerboard()

        loss = EstimateExtrinsicsLoss(cb, frame_data, self._config.huber_loss_delta)

        res = least_squares(
            loss,
            np.ones(loss.n_params),
            x_scale='jac',
            ftol=self._config.optimization_tolerance,
            method='trf')

        transforms = loss.decompose_params(res.x.copy())
        result = dict()
        for id, t in transforms.items():
            result[id] = t.as_matrix()
        return result

    def create_env_mesh(self, frames, cam_id, calibration: RobotCalibration):
        """
        Integrates all frames into single mesh

        Args:
            frames: frames collected with 'collect_frames' method
            intrinsics: intrinsic camera parameters, found with 'calibrate' method
            camera_to_tcp: calibrated transformation matrix from camera system to TCP system, found with 'calibrate' method

        Returns:
            mesh (open3d.geometry.TriangleMesh): computed mesh in robot coordinate system
        """

        intrinsics = calibration.gripper_camera.depth_intrinsics
        camera_to_tcp = calibration.gripper_camera_to_tcp

        model = _SLAMModel(self._config.slam_model, intrinsics, frames[0][cam_id].infrared)

        print('Integrating frames into single volume...')
        for frame in tqdm(frames):
            model.integrate(frame[cam_id].depth, frame[cam_id].infrared)

        mesh, poses = model.extract_results()

        mat = np.zeros((4, 4))
        n_frames = 0

        for pose, robot_pos in zip(poses, [f['robot_pos'] for f in frames]):
            p = pose.copy()
            p[:3, 3] *= 1000
            mat += compose_matrix_world(robot_pos[:3], robot_pos[3:]) @ camera_to_tcp @ np.linalg.inv(p)
            n_frames += 1

        u, _, vt = np.linalg.svd(mat[:3, :3], full_matrices=True)
        mat[:3, :3] = u @ vt
        mat[:3, 3] /= n_frames * 1000
        mat[3, 3] = 1

        mesh.transform(mat)

        return mesh

    def _get_random_robot_position(self):
        r = 2 * (np.random.rand(3) - 0.5)
        x, y, z = (r * np.array(self._config.max_shifts))
        roll, pitch, yaw = 0, 0, 0

        max_roll, max_pitch, max_yaw = self._config.max_angles

        roll = 2 * (np.random.rand(1).item() - 0.5) * max_roll
        pitch = 2 * (np.random.rand(1).item() - 0.5) * max_pitch
        yaw = 2 * (np.random.rand(1).item() - 0.5) * max_yaw

        res = np.array([x.item(), y.item(), z.item(), roll, pitch, yaw])
        return res + np.array(self._config.home_pos)

    def _extract_data_for_calibration(self, frames):
        result = defaultdict(list)

        for i, frame in enumerate(tqdm(frames)):
            for id, cam in frame.items():
                if id == 'robot_pos':
                    continue

                ir_corners, depths, rgb_corners = self._get_checkerboard_points(cam)
                if ir_corners is None:
                    continue

                f = _FrameData(
                    frame_index=i,
                    points=ir_corners,
                    depths=depths,
                    rgb_points=rgb_corners)

                if 'robot_pos' in frame:
                    f.robot_pos = frame['robot_pos']

                result[id].append(f)

        return result

    def _get_checkerboard_points(self, frame):
        infrared_found, infrared_corners = cv2.findChessboardCorners(
            frame.infrared,
            self._config.checkerboard_size)

        if not infrared_found:
            return None, None, None

        infrared_corners = infrared_corners.astype(int).squeeze()
        d = frame.depth[infrared_corners[:, 1], infrared_corners[:, 0]]
        if not np.all(d > 0):
            return None, None, None

        found_rgb, corners_rgb = cv2.findChessboardCorners(
            cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY),
            self._config.checkerboard_size)

        if found_rgb:
            corners_rgb = corners_rgb.astype(int).squeeze()

        return infrared_corners, d, corners_rgb

    def _estimate_boards_positions(self, sensor, frame_data):
        cb = self._make_checkerboard()

        points = np.stack([f.points for f in frame_data])
        depths = np.stack([f.depths for f in frame_data])

        loss = EstimateBoardPositionsLoss(cb, points, depths, sensor, self._config.huber_loss_delta)

        res = least_squares(
            loss,
            np.ones(loss.n_params),
            jac_sparsity=loss.get_jacobian_sparsity(),
            x_scale='jac',
            ftol=self._config.optimization_tolerance,
            method='trf')

        return loss.decompose_params(res.x)['transforms']

    def _calibrate_depth_parameters(self, cb, sensor_init, frame_data):
        sensor = Sensor(sensor_init)
        board_positions = self._estimate_boards_positions(sensor, frame_data)

        points = np.stack([f.points for f in frame_data])
        depths = np.stack([f.depths for f in frame_data])
        loss = EstimateIntrinsicsLoss(cb, points, depths, self._config.huber_loss_delta)

        res = least_squares(
            loss,
            np.concatenate([sensor_init] + [t.params for t in board_positions]),
            jac_sparsity=loss.get_jacobian_sparsity(),
            x_scale='jac',
            ftol=self._config.optimization_tolerance,
            method='trf')

        params = loss.decompose_params(res.x.copy())

        for f, board_pos in zip(frame_data, params['transforms']):
            f.board_pos = board_pos

        return params['sensor']

    def _calibrate_color_parameters(self, cb, sensor_init, frame_data):
        sensor = Sensor(sensor_init)

        color_points = [f.rgb_points for f in frame_data]
        board_positions = [f.board_pos for f in frame_data]
        loss = EstimateColorParametersLoss(cb, board_positions, color_points, self._config.huber_loss_delta)
        color_sensor = Sensor(sensor_init)

        res = least_squares(
            loss,
            np.concatenate([sensor_init, np.zeros(6)]),
            x_scale='jac',
            ftol=self._config.optimization_tolerance,
            method='trf')

        params = loss.decompose_params(res.x)
        return params['sensor'], params['transform']

    def _calibrate_camera_to_tcp(self, frame_data):
        cb = self._make_checkerboard()

        board_positions = [f.board_pos for f in frame_data]
        positions = [f.robot_pos for f in frame_data]

        loss = EstimateGlobalBoardPositionLoss(cb, positions, board_positions, self._config.huber_loss_delta)
        res = least_squares(
            loss, np.ones(loss.n_params),
            x_scale='jac',
            ftol=self._config.optimization_tolerance,
            method='trf')

        return loss.decompose_params(res.x)['camera_to_tcp'].as_matrix()

    def _transform_extrinsics(self, extrinsics, frame_data, gripper_camera_to_tcp, gripper_camera_id):
        robot_pos = frame_data[gripper_camera_id][0].robot_pos
        tcp_to_base = compose_matrix_world(robot_pos[:3], robot_pos[3:])
        gripper_camera_to_base = tcp_to_base @ gripper_camera_to_tcp

        result = dict()
        tf = gripper_camera_to_base @ np.linalg.inv(extrinsics[gripper_camera_id])

        for id, extr in extrinsics.items():
            result[id] = tf @ extr

        return result

    def _make_checkerboard(self):
        return Checkerboard(
            self._config.checkerboard_size,
            self._config.checkerboard_cell_size)


@dataclass
class _FrameData:
    frame_index: int

    points: NDArray = None
    depths: NDArray = None
    rgb_points: NDArray = None

    robot_pos: NDArray = None
    board_pos: Transformation = None


class _SLAMModel:
    def __init__(self, config, intrinsic, sample_frame):
        self._config = config
        self._device = o3d.core.Device(config.device)

        h, w = sample_frame.shape[:2]

        intrinsic_o3d = o3d.core.Tensor(intrinsic)
        self._input_frame = o3d.t.pipelines.slam.Frame(h, w, intrinsic_o3d, self._device)
        self._raycast_frame = o3d.t.pipelines.slam.Frame(h, w, intrinsic_o3d, self._device)

        self._frame_to_model = o3d.core.Tensor(np.identity(4))

        self._model = o3d.t.pipelines.slam.Model(
            config.voxel_size,
            config.block_resolution,
            config.block_count,
            self._frame_to_model,
            self._device)

        self._poses = []

    def integrate(self, depth, infrared):
        self._input_frame.set_data_from_image(
            'depth',
            o3d.t.geometry.Image(depth.astype(np.float32) / self._config.depth_scale).to(self._device))

        self._input_frame.set_data_from_image(
            'color',
            o3d.t.geometry.Image(np.dstack([infrared, infrared, infrared]).astype(np.float32) / 255).to(self._device))

        if len(self._poses) > 0:
            result = self._model.track_frame_to_model(
                self._input_frame,
                self._raycast_frame,
                1.0,
                self._config.depth_max,
                self._config.odometry_distance_threshold)
            self._frame_to_model = self._frame_to_model @ result.transformation

        self._model.update_frame_pose(len(self._poses), self._frame_to_model)
        self._poses.append(self._frame_to_model.cpu().numpy())

        self._model.integrate(
            self._input_frame,
            1.0,
            self._config.depth_max,
            self._config.trunc_voxel_multiplier)

        self._model.synthesize_model_frame(
            self._raycast_frame,
            1.0,
            self._config.depth_min,
            self._config.depth_max,
            self._config.trunc_voxel_multiplier,
            False)

    def extract_results(self):
        return self._model.extract_trianglemesh().to_legacy(), np.stack(self._poses)

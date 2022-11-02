from ruka.robot.xarm_ import XArmPosControlled, ControlMode
import time
import numpy as np
import cv2
from tqdm.auto import tqdm
from scipy.optimize import least_squares
from ruka.cv.calibration.loss import EstimateBoardPositionsLoss, EstimateIntrinsicsLoss, EstimateGlobalBoardPositionLoss
from ruka.cv.calibration import Checkerboard, Sensor, Transformation
import dataclasses
import open3d as o3d
from ruka.util.x3d import compose_matrix_world
from typing import Tuple
from ruka.robot.realsense import RealsenseCamera


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

    n_frames: int
    save_every_frames: int
    max_shifts: Tuple[float, float, float]
    max_angles: Tuple[float, float, float]
    home_pos: np.array
    checkerboard_size: Tuple[int, int]
    checkerboard_cell_size: float
    optimization_tolerance: float
    huber_loss_delta: float
    focal_length_init_value: float

    slam_model: SLAMModelConfig


class Calibrator:
    def __init__(self, config: CalibratorConfig):
        self._config = config

    def collect_frames(self, robot_controller: XArmPosControlled, camera: RealsenseCamera):
        """
        Collects frames for calibration using random robot movement

        Args:
            robot_controller: position controller to use
            camera: camera to capture frames from

        Returns:
            frames (List[dict]): list of collected frames
        """

        frames = []
        frames_cnt = 0
        def collect_frame():
            nonlocal frames_cnt, frames, robot_controller

            f = camera.capture()
            if frames_cnt % self._config.save_every_frames == 0:
                pos = np.array(robot_controller.pos + robot_controller.angles)
                pos[-2:] *= -1 # TODO: remove when contoller follows conventions

                frames.append(dict({
                    'depth': f[:, :, 0].astype(np.uint16),
                    'infrared': f[:, :, 1].astype(np.uint8),
                    'pos': pos,
                }))
            frames_cnt += 1

        robot_controller.go_home()
        robot_controller.steady(ControlMode.POS)

        while len(frames) < self._config.n_frames:
            print('Collected {}/{} frames'.format(len(frames), self._config.n_frames))

            try:
                target = self._get_random_robot_position()
                robot_controller.set_pos(target[:3], target[3:])
                while not robot_controller.is_target_reached():
                    collect_frame()
            except:
                robot_controller.go_home()
                robot_controller.steady(ControlMode.POS)

        return frames

    def calibrate(self, frames):
        """
        Calibrates intrinsic and extrinsic camera parameters

        Args:
            frames: frames collected with 'collect_frames' method

        Returns:
            intrinsics (np.array): camera intrinsic calibration matrix
            camera_to_tcp (np.array): calibrated transformation matrix from camera system to TCP system
        """

        points, depths, positions = self._extract_data_for_calibration(frames)

        cb = Checkerboard(self._config.checkerboard_size, self._config.checkerboard_cell_size)
        sensor_init = np.array([
            self._config.focal_length_init_value,
            self._config.focal_length_init_value,
            frames[0]['infrared'].shape[1] // 2,
            frames[0]['infrared'].shape[0] // 2
        ])

        sensor = Sensor(sensor_init)
        loss = EstimateBoardPositionsLoss(cb, points, depths, sensor, self._config.huber_loss_delta)

        print('Estimating board positions...')
        res = least_squares(
            loss,
            np.ones(loss.n_params),
            jac_sparsity=loss.get_jacobian_sparsity(),
            x_scale='jac',
            ftol=self._config.optimization_tolerance,
            method='trf')

        loss = EstimateIntrinsicsLoss(cb, points, depths, self._config.huber_loss_delta)

        print('Estimating intrinsics...')
        res = least_squares(
            loss,
            np.concatenate([sensor_init, res.x]),
            jac_sparsity=loss.get_jacobian_sparsity(),
            x_scale='jac',
            ftol=self._config.optimization_tolerance,
            method='trf')

        params = loss.decompose_params(res.x.copy())

        print('Estimating board and camera positions')
        loss = EstimateGlobalBoardPositionLoss(cb, positions, params['transforms'], self._config.huber_loss_delta)
        res = least_squares(
            loss, np.ones(loss.n_params),
            x_scale='jac',
            ftol=self._config.optimization_tolerance,
            method='trf')

        camera_to_tcp = loss.decompose_params(res.x)['camera_to_tcp'].as_matrix()
        intrinsics = params['sensor'].intrinsics_as_matrix()

        return intrinsics, camera_to_tcp

    def create_env_mesh(self, frames, intrinsics, camera_to_tcp):
        """
        Integrates all frames into single mesh

        Args:
            frames: frames collected with 'collect_frames' method
            intrinsics: intrinsic camera parameters, found with 'calibrate' method
            camera_to_tcp: calibrated transformation matrix from camera system to TCP system, found with 'calibrate' method

        Returns:
            mesh (open3d.geometry.TriangleMesh): computed mesh in robot coordinate system
        """

        model = _SLAMModel(self._config.slam_model, intrinsics, frames[0]['infrared'])

        print('Integrating frames into single volume...')
        for frame in tqdm(frames):
            model.integrate(frame['depth'], frame['infrared'])

        mesh, poses = model.extract_results()

        mat = np.zeros((4, 4))
        n_frames = 0

        for pose, robot_pos in zip(poses, [f['pos'] for f in frames]):
            p = pose.copy()
            p[:3, 3] *= 1000
            mat += compose_matrix_world(robot_pos[:3], robot_pos[3:]) @ camera_to_tcp @ np.linalg.inv(p)
            n_frames += 1

        np.set_printoptions(precision=3, floatmode='fixed', suppress=True)

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
        return res + self._config.home_pos

    def _extract_data_for_calibration(self, frames):
        points = []
        depths = []
        positions = []

        print('Looking for checkerboards...')

        for i, frame in enumerate(tqdm(frames)):
            img = frame['infrared']
            depth = frame['depth']
            found, corners = cv2.findChessboardCorners(img, self._config.checkerboard_size)

            if found:
                corners = corners.astype(int).squeeze()
                d = depth[corners[:, 1], corners[:, 0]]

                if not np.all(d > 0):
                    continue

                points.append(corners)
                depths.append(d)
                positions.append(np.array(frame['pos']))

        points = np.stack(points)
        depths = np.stack(depths)
        positions = np.stack(positions)

        return points, depths, positions


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

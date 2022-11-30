import numpy as np

from ruka.util.x3d import compose_matrix_world, compose_matrix_tool
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R
from scipy.special import huber

from .sensor import Sensor
from .transformation import Transformation


class BaseLoss:
    def __call__(self, x):
        raise NotImplementedError()

    def get_jacobian_sparsity(self):
        raise NotImplementedError()

    @property
    def n_params(self):
        raise NotImplementedError()

    @property
    def n_residuals(self):
        raise NotImplementedError()

    def decompose_params(self, x) -> dict:
        raise NotImplementedError()


class EstimateBoardPositionsLoss(BaseLoss):
    def __init__(self, checkerboard, points, depths, sensor, huber_loss_delta):
        super().__init__()

        assert len(points.shape) == 3
        assert points.shape[2] == 2
        assert depths.shape[:2] == points.shape[:2]

        self.points_per_board = checkerboard.size[0] * checkerboard.size[1]
        self.n_boards = points.shape[0]

        self.points = points.reshape(-1, 2)
        self.depths = depths.reshape(-1)
        self.sensor = sensor

        self.cb_points = np.concatenate([checkerboard.get_points()] * points.shape[0], axis=0)

        self.huber_loss_delta = huber_loss_delta

    def __call__(self, x):
        pts = self.sensor.deproject(self.points, self.depths)
        cb_pts = self._transform_cb(x)

        return np.sqrt(huber(self.huber_loss_delta, (pts - cb_pts).reshape(-1)))

    def get_jacobian_sparsity(self):
        res = lil_matrix((self.n_residuals, self.n_params,), dtype=int)

        for i in range(self.n_boards):
            res[
                (3 * self.points_per_board * i):(3 * self.points_per_board * (i + 1)),
                (Transformation.N_PARAMS * i):(Transformation.N_PARAMS * (i + 1))
            ] = 1

        return res

    @property
    def n_params(self):
        return Transformation.N_PARAMS * self.n_boards

    @property
    def n_residuals(self):
        return 3 * self.points.shape[0]

    def _transform_cb(self, x):
        transforms = [
            Transformation(x[offset:(offset + Transformation.N_PARAMS)])
            for offset in range(0, x.size, Transformation.N_PARAMS)
        ]

        res = []
        for i, tf in enumerate(transforms):
            res.append(tf.transform(self.cb_points[(self.points_per_board * i):((i + 1) * self.points_per_board)]))
        return np.concatenate(res, axis=0)


class EstimateIntrinsicsLoss(BaseLoss):
    def __init__(self, checkerboard, points, depths, huber_loss_delta):
        super().__init__()

        assert len(points.shape) == 3
        assert points.shape[2] == 2
        assert depths.shape[:2] == points.shape[:2]

        self.points_per_board = checkerboard.size[0] * checkerboard.size[1]
        self.n_boards = points.shape[0]

        self.points = points.reshape(-1, 2)
        self.depths = depths.reshape(-1)

        self.cb_points = np.concatenate([checkerboard.get_points()] * points.shape[0], axis=0)

        self.huber_loss_delta = huber_loss_delta

    def __call__(self, x):
        params = self.decompose_params(x)

        pts = params['sensor'].deproject(self.points, self.depths)
        cb_pts = self._transform_cb(params['transforms'])

        return np.sqrt(huber(self.huber_loss_delta, (pts - cb_pts).reshape(-1)))

    def get_jacobian_sparsity(self):
        res = lil_matrix((self.n_residuals, self.n_params,), dtype=int)

        for i in range(self.n_boards):
            res[
                (3 * self.points_per_board * i):(3 * self.points_per_board * (i + 1)),
                (Sensor.N_PARAMS + Transformation.N_PARAMS * i):(Sensor.N_PARAMS + Transformation.N_PARAMS * (i + 1))
            ] = 1
            res[:, :Sensor.N_PARAMS] = 1

        return res

    @property
    def n_params(self):
        return Sensor.N_PARAMS + Transformation.N_PARAMS * self.n_boards

    @property
    def n_residuals(self):
        return 3 * self.points.shape[0]

    def decompose_params(self, x) -> dict:
        sensor = Sensor(x[:Sensor.N_PARAMS])

        transforms = [
            Transformation(x[offset:(offset + Transformation.N_PARAMS)])
            for offset in range(Sensor.N_PARAMS, x.size, Transformation.N_PARAMS)
        ]

        return {
            'sensor': sensor,
            'transforms': transforms,
        }

    def _transform_cb(self, transforms):
        res = []
        for i, tf in enumerate(transforms):
            res.append(tf.transform(self.cb_points[(self.points_per_board * i):((i + 1) * self.points_per_board)]))
        return np.concatenate(res, axis=0)


class EstimateGlobalBoardPositionLoss(BaseLoss):
    def __init__(self, checkerboard, tcp_positions, board_positions, huber_loss_delta):
        super().__init__()

        self.tcp_positions = [compose_matrix_world(pos[:3], pos[3:]) for pos in tcp_positions]
        self.cb_points_camera_system = [tf.transform(checkerboard.get_points()) for tf in board_positions]
        self.cb_points = checkerboard.get_points()

        self.huber_loss_delta = huber_loss_delta

    def __call__(self, x):
        base_to_board = Transformation(x[:Transformation.N_PARAMS])
        camera_to_tcp = Transformation(x[Transformation.N_PARAMS:])

        res = []
        for base_to_tcp, cb_points in zip(self.tcp_positions, self.cb_points_camera_system):
            pts = camera_to_tcp.transform(cb_points)

            pts = pts @ base_to_tcp[:3, :3].T
            pts += base_to_tcp[:3, 3][None]

            pts = base_to_board.transform(pts)

            res.append(pts - self.cb_points)

        res = np.concatenate(res, axis=0)
        return np.sqrt(huber(self.huber_loss_delta, res.reshape(-1)))

    @property
    def n_params(self):
        return 2 * Transformation.N_PARAMS

    @property
    def n_residuals(self):
        return 3 * self.points.shape[0]

    def decompose_params(self, x) -> dict:
        return {
            'camera_to_tcp': Transformation(x[Transformation.N_PARAMS:]),
        }


class EstimateColorParametersLoss(BaseLoss):
    def __init__(
        self,
        checkerboard,
        board_positions,
        color_points,
        huber_loss_delta):

        super().__init__()

        self.board_points = []
        self.points = []
        for points, transform in zip(color_points, board_positions):
            if points is None:
                continue
            self.board_points.append(transform.transform(checkerboard.get_points()))
            self.points.append(points)

        self.board_points = np.stack(self.board_points)
        self.points = np.stack(self.points)

        self.huber_loss_delta = huber_loss_delta

    def __call__(self, x):
        params = self.decompose_params(x)

        residuals = []
        for points, board_points in zip(self.points, self.board_points):
            board_transformed = params['transform'].transform(board_points)
            projected = params['sensor'].project(board_transformed)

            residuals.append((points - projected).reshape(-1))

        return np.sqrt(huber(self.huber_loss_delta, np.concatenate(residuals)))

    @property
    def n_params(self):
        return Sensor.N_PARAMS + Transformation.N_PARAMS

    @property
    def n_residuals(self):
        return 2 * self.points.shape[0] * self.points.shape[1]

    def decompose_params(self, x) -> dict:
        sensor = Sensor(x[:Sensor.N_PARAMS])
        transform = Transformation(x[Sensor.N_PARAMS:])

        return {
            'sensor': sensor,
            'transform': transform,
        }

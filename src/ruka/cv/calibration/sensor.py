import numpy as np

class Sensor:
    N_PARAMS = 4
    FX_INDEX = 0
    FY_INDEX = 1
    PPX_INDEX = 2
    PPY_INDEX = 3

    def __init__(self, params):
        assert params.size == self.N_PARAMS
        self.params = params

    def deproject(self, pts, depths):
        fx, fy, ppx, ppy = self.params

        z = np.ones(pts.shape[0]) * depths
        x = ((pts[:, 0] - ppx) / fx) * z
        y = ((pts[:, 1] - ppy) / fy) * z

        return np.stack([x, y, z]).T

    def project(self, pts):
        x, y, z = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2]
        x = ((x / z) * self.params[self.FX_INDEX]) + self.params[self.PPX_INDEX]
        y = ((y / z) * self.params[self.FY_INDEX]) + self.params[self.PPY_INDEX]

        return np.vstack([x, y]).T

    def intrinsics_as_matrix(self):
        return np.array([
            [self.params[self.FX_INDEX], 0, self.params[self.PPX_INDEX]],
            [0, self.params[self.FY_INDEX], self.params[self.PPY_INDEX]],
            [0, 0, 1],
        ])

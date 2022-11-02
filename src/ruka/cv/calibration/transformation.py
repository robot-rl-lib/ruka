import numpy as np
from scipy.spatial.transform import Rotation as R


class Transformation:
    N_PARAMS = 6

    def __init__(self, params):
        assert params.size == self.N_PARAMS

        self.params = params

    def transform(self, pts):
        mat = R.from_rotvec(self.params[:3]).as_matrix()
        return (pts @ mat.T) + self.params[3:][None]

    def as_matrix(self):
        mat = np.eye(4)
        mat[:3, :3] = R.from_rotvec(self.params[:3]).as_matrix()
        mat[:3, 3] = self.params[3:]
        return mat

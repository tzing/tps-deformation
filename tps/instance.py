import numpy

from . import functions

__all__ = ['TPS']


class TPS:
    """
    The thin plate spline deformation warpping
    """

    def __init__(self,
                 control_points: numpy.ndarray,
                 target_points: numpy.ndarray,
                 lambda_: float = 0.,
                 solver: str = 'exact'):
        """Create a instance that preserve the TPS coefficients.

        Args:
            control_points (numpy.array): p by d vector of control points.
            target_points (numpy.array): p by d vector of corresponding control
                points in the mapping function f(S).
            lambda_ (float): regularization parameter. See page 4 in reference.
        """
        self.control_points = control_points
        self.coefficient = functions.find_coefficients(
            control_points, target_points, lambda_, solver)

    def __call__(self, source_points):
        """Transform the source points form the original surface to the
        destination (deformed) surface.

        Args:
            source_points (numpy.array): n by 3 array of X, Y, Z components of
                the surface.
            this funcaiton i
        """
        return functions.transform(source_points, self.control_points,
                                   self.coefficient)

    transform = __call__

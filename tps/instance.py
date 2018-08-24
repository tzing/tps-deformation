import numpy

from . import functions

__all__ = ['TPS']


class TPS:
    """The thin plate spline deformation warpping.
    """

    def __init__(self,
                 control_points: numpy.ndarray,
                 target_points: numpy.ndarray,
                 lambda_: float = 0.,
                 solver: str = 'exact'):
        """Create a instance that preserve the TPS coefficients.

        Arguments
        ---------
            control_points : numpy.array
                p by d vector of control points
            target_points : numpy.array
                p by d vector of corresponding target points on the deformed
                surface
            lambda_ : float
                regularization parameter
            solver : str
                the solver to get the coefficients. default is 'exact' for the
                exact solution. Or use 'lstsq' for the least square solution.
        """
        self.control_points = control_points
        self.coefficient = functions.find_coefficients(
            control_points, target_points, lambda_, solver)

    def __call__(self, source_points):
        """Transform the source points form the original surface to the
        destination (deformed) surface.

        Arguments
        ---------
            source_points : numpy.array
                n by d array of source points to be transformed
        """
        return functions.transform(source_points, self.control_points,
                                   self.coefficient)

    transform = __call__

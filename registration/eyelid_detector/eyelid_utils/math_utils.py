import cv2
import numpy as np

__all__ = ['ellipse_correction']


def ellipse_correction(points):
    ellipse_param = cv2.fitEllipse(points)
    resampled_points = ellipse_resampling(ellipse_param, points)

    return resampled_points


def ellipse_resampling(ellipse_param, points):
    """
    Given ellipse parameters and raw points, return resampled points
    Args:
        ellipse_param: (center_x, center_y), (2a, 2b), radians_angle
    """

    ellipse_center, _, _ = ellipse_param

    new_points = []
    for point in points:
        line_points = [ellipse_center, point]
        new_point = line_ellipse_intersection(line_points, ellipse_param)
        new_points.append(new_point)
    return np.array(new_points)

def line_ellipse_intersection(line_points, ellipse_param):
    """
    Given a line and a ellipse
    Args:
        line_points: two points that determines the line
        ellipse_param: (center_x, center_y), (2a, 2b), radians_angle
    """

    line_point1 = line_points[0]    # ellipse_center
    line_point2 = line_points[1]

    ellipse_center, double_axes, radians_angle = ellipse_param
    ellipse_axes = (double_axes[0] / 2, double_axes[1] / 2)
    ellipse_angle = np.radians(radians_angle)

    ############## MODIFIED SECTION ############
    # Rotate the line back to an unrotated ellipse
    theta = np.arctan2( line_point2[1] - line_point1[1], line_point2[0] - line_point1[0] ) - ellipse_angle
    m = np.tan( theta )
    c = line_point1[1] - m * line_point1[0]
    a, b, h, k = ellipse_axes[0], ellipse_axes[1], ellipse_center[0], ellipse_center[1]
    A = ( a * m ) ** 2 + b ** 2
    B = 2 * ( a ** 2 * m * ( c - k ) - b ** 2 * h )
    C = ( b * h ) ** 2 + ( a * ( c - k ) ) ** 2 - ( a * b ) ** 2
    discriminant = B ** 2 - 4 * A * C
    # Intersection of back-rotated line with back-rotated ellipse (need correct root)
    theta = np.mod( theta, 2 * np.pi )
    x1 = ( -B + np.sqrt( discriminant ) ) / ( 2 * A )                 # quadratic solution (larger root)
    if theta > 0.5 * np.pi and theta < 1.5 * np.pi: x1 = -B/A - x1    # need other root
    y1 = m * x1 + c
    # Rotate the line forward again
    x1, y1 = h + ( x1 - h ) * np.cos( ellipse_angle ) - ( y1 - k ) * np.sin( ellipse_angle ),  k + ( x1 - h ) * np.sin( ellipse_angle ) + ( y1 - k ) * np.cos( ellipse_angle )
    intersection_point = (int(x1), int(y1))
    ############## END OF MODIFIED SECTION ############

    return intersection_point

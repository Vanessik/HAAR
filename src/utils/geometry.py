import numpy as np
import torch
from matplotlib.path import Path


def barycentric_coordinates(p, a, b, c):
    """
    Compute the barycentric coordinates of point p with respect to the triangle defined by points a, b, and c.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def map_uv_to_3d(U, V, faces, uv_map, vertices):
    """
    Map 2D UV meshgrid points to 3D positions and face index using barycentric interpolation with NumPy.
    Return a tensor of the same shape as the meshgrid with 4 channels (3 for position, 1 for face index).
    """
    # Initialize a tensor for 3D positions with a default value
    positions_tensor = torch.full((U.shape[0], U.shape[1], 4), -100.0, dtype=torch.float32)
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            uv_point = np.array([U[i, j], V[i, j]])
            for idx, face in enumerate(faces):
                triangle_uv = uv_map[0][face]
                path = Path(triangle_uv)
                if path.contains_point(uv_point):
                    # Calculate barycentric coordinates
                    u, v, w = barycentric_coordinates(uv_point, *triangle_uv)
                    if 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1:
                        # Calculate 3D position using barycentric interpolation
                        triangle_3d = vertices[face]
                        position_3d = u * triangle_3d[0] + v * triangle_3d[1] + w * triangle_3d[2]
                        positions_tensor[i, j, :3] = torch.tensor(position_3d)
                        positions_tensor[i, j, 3] = idx
                        break
                
    return positions_tensor


def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    #(p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    p = points
    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    return weights


def check_barycentric_weights(barycentric_weights):
#     return indexes of correct barycentric weights

    a = torch.nonzero(barycentric_weights[:, 0] >= 0).squeeze(-1).cpu().numpy()
    b = torch.nonzero(barycentric_weights[:, 1] >= 0).squeeze(-1).cpu().numpy()
    c =  torch.nonzero(barycentric_weights[:, 2] >= 0).squeeze(-1).cpu().numpy()
    return np.intersect1d(np.intersect1d(a, b), c)

"""Copied and adapted from hloc
3D visualization based on plotly.
Works for a small number of points and cameras, might be slow otherwise.

1) Initialize a figure with `init_figure`
2) Add 3D points, camera frustums, or both as a pycolmap.Reconstruction

Written by Paul-Edouard Sarlin and Philipp Lindenberger.
"""

from typing import Optional

import numpy as np
import plotly.graph_objects as go
import pycolmap


def to_homogeneous(points):
    pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)


def init_figure(height: int = 1000) -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )
    fig.update_layout(
        # template="plotly_dark",
        # use white background
        template="plotly",
        height=height,
        scene_camera=dict(
            eye=dict(x=0.0, y=-0.1, z=-2),
            up=dict(x=0, y=-1.0, z=0),
            projection=dict(type="orthographic"),
        ),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode="data",
            # aspectratio=dict(x=1, y=1, z=1),
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.1),
    )
    return fig


def plot_points(
    fig: go.Figure,
    pts: np.ndarray,
    color: str = "rgba(255, 0, 0, 1)",
    ps: int = 2,
    colorscale: Optional[str] = None,
    name: Optional[str] = None,
    showlegend: bool = True,
):
    """Plot a set of 3D points."""
    x, y, z = pts.T
    tr = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        name=name,
        legendgroup=name,
        showlegend=showlegend,
        marker=dict(size=ps, color=color, line_width=0.0, colorscale=colorscale),
    )
    fig.add_trace(tr)


def plot_camera(
    fig: go.Figure,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    legendgroup: Optional[str] = None,
    show_legend: bool = True,
    fill: bool = False,
    size: float = 1.0,
    text: Optional[str] = None,
    width=10,
    **kwargs,
):
    """Plot a camera frustum from pose and intrinsic matrix."""
    W, H = K[0, 2] * 2, K[1, 2] * 2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    if size is not None:
        image_extent = max(size * W / 1024.0, size * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        scale = 0.5 * image_extent / world_extent
    else:
        scale = 1.0
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners / 2 * scale) @ R.T + t
    legendgroup = legendgroup if legendgroup is not None else name
    x, y, z = np.concatenate(([t], corners)).T
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    if fill:
        name = None
        pyramid = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            color=color,
            i=i,
            j=j,
            k=k,
            legendgroup=legendgroup,
            name=name,
            showlegend=True,
            hovertemplate=text.replace("\n", "<br>"),
        )
        fig.add_trace(pyramid)

    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))
    tri_points = np.array([vertices[i] for i in triangles.reshape(-1)])
    x, y, z = tri_points.T
    pyramid = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        legendgroup=legendgroup,
        name=name,
        line=dict(color=color, width=width),
        showlegend=show_legend,
        hovertemplate=text.replace("\n", "<br>"),
    )

    fig.add_trace(pyramid)


def plot_camera_colmap(
    fig: go.Figure,
    image: pycolmap.Image,
    camera: pycolmap.Camera,
    name: Optional[str] = None,
    show_legend: bool = False,
    legendgroup: Optional[str] = None,
    **kwargs,
):
    """Plot a camera frustum from PyCOLMAP objects"""
    world_t_camera = image.cam_from_world.inverse()
    if name is None:
        name = image.name
    plot_camera(
        fig,
        world_t_camera.rotation.matrix(),
        world_t_camera.translation,
        camera.calibration_matrix(),
        # name=name or str(image.image_id),
        name=name,
        legendgroup=name,
        text=str(image),
        show_legend=show_legend,
        **kwargs,
    )


def plot_cameras(fig: go.Figure, reconstruction: pycolmap.Reconstruction, **kwargs):
    """Plot a camera as a cone with camera frustum."""
    show_legend = kwargs.get("name") is not None
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        plot_camera_colmap(fig, image, reconstruction.cameras[image.camera_id], show_legend=show_legend, **kwargs)
        show_legend = False


def plot_reconstruction(
    fig: go.Figure,
    rec: pycolmap.Reconstruction,
    max_reproj_error: float = 300000,
    color: str = "rgb(0, 0, 255)",
    cam_color: str = "rgb(255, 0, 0)",
    name: Optional[str] = None,
    min_track_length: int = 2,
    points: bool = True,
    cameras: bool = True,
    points_rgb: bool = True,
    cs: float = 1.0,
    ps: int = 1,
    names=None,
    **kwargs,
):
    bbs = rec.compute_bounding_box(0.001, 0.999)
    # Filter points, use original reproj error here
    p3Ds = [
        p3D
        for _, p3D in rec.points3D.items()
        if (
            (p3D.xyz >= bbs.min).all()
            and (p3D.xyz <= bbs.max).all()
            and p3D.error <= max_reproj_error
            and p3D.track.length() >= min_track_length
        )
    ]
    if len(p3Ds) > 0:
        xyzs = [p3D.xyz for p3D in p3Ds]
        pcolor = [p3D.color for p3D in p3Ds] if points_rgb else color
        if points:
            plot_points(fig, np.array(xyzs), color=pcolor, ps=ps, name=f"{name}_points")
    if cameras:
        plot_cameras(fig, rec, legendgroup=f"{name}_cameras", size=cs, color=cam_color, **kwargs)

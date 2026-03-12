"""copied and adapted from hloc
2D visualization primitives based on Matplotlib.

1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


def plot_images(
    imgs,
    titles=None,
    cmaps="gray",
    dpi=100,
    pad=0.5,
    adaptive=True,
    figsize=4.5,
    top=None,
    bottom=None,
    left=None,
    right=None,
    bad="white",
    vmax=None,
    **kwargs,
):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    ratios = [i.shape[1] / i.shape[0] for i in imgs] if adaptive else [4 / 3] * n
    figsize = [sum(ratios) * figsize, figsize]
    fig, axs = plt.subplots(1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios})
    if n == 1:
        axs = [axs]
    for i, (img, ax) in enumerate(zip(imgs, axs)):
        cmap = plt.get_cmap(cmaps[i]).copy()
        cmap.set_bad(bad)
        ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_axis_off()
        if titles:
            ax.set_title(titles[i])
    fig.tight_layout(pad=pad)
    fig.subplots_adjust(
        top=top, bottom=bottom, left=left, right=right
    )  # Adjust the top padding (set it between 0 and 1)
    return fig, axs


def fig_to_numpy_array(fig):
    """Convert a Matplotlib figure to a NumPy array."""
    fig.canvas.draw()  # Render the figure
    buf = fig.canvas.tostring_argb()  # Get buffer in ARGB format
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)  # Reshape to an image
    img = img[:, :, [1, 2, 3, 0]]  # Reorder ARGB -> RGBA
    return img[:, :, :3]


def plot_keypoints(kpts, colors="lime", ps=4, alpha=1, **kwargs):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    axes = plt.gcf().axes
    for a, k, c in zip(axes, kpts, colors):
        if len(k) == 0:
            c = "r"
        a.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha)


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, indices=(0, 1), a=1.0):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        for i in range(len(kpts0)):
            fig.add_artist(
                matplotlib.patches.ConnectionPatch(
                    xyA=(kpts0[i, 0], kpts0[i, 1]),
                    coordsA=ax0.transData,
                    xyB=(kpts1[i, 0], kpts1[i, 1]),
                    coordsB=ax1.transData,
                    zorder=1,
                    color=color[i],
                    linewidth=lw,
                    alpha=a,
                )
            )

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps, alpha=1)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps, alpha=1)


def add_text(
    idx,
    text,
    pos=(0.01, 0.99),
    fs=15,
    color="w",
    lcolor="k",
    lwidth=2,
    ha="left",
    va="top",
):
    ax = plt.gcf().axes[idx]
    t = ax.text(*pos, text, fontsize=fs, ha=ha, va=va, color=color, transform=ax.transAxes)
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)


def vis_dense_overlap(parser, imid1, imid2, overlap_kwargs):
    comb_mask_21 = overlap_kwargs["occlusion_mask_21"] & overlap_kwargs["valid_mask_21"]
    comb_mask_12 = overlap_kwargs["occlusion_mask_12"] & overlap_kwargs["valid_mask_12"]
    p2D1 = np.where(overlap_kwargs["valid1_mask"])
    p2D2 = np.where(overlap_kwargs["valid2_mask"])

    # set size of fig
    rgb1 = parser.rgb(imid1)
    rgb2 = parser.rgb(imid2)
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(15, 12)
    axs[0, 0].imshow(rgb1)
    axs[0, 0].scatter(p2D1[1], p2D1[0], c="r", s=0.005, alpha=0.3)
    axs[0, 1].imshow(rgb2)
    axs[0, 1].scatter(p2D2[1], p2D2[0], c="r", s=0.005, alpha=0.3)
    axs[1, 0].imshow(rgb1)
    axs[1, 0].scatter(
        overlap_kwargs["p2D21"][:, 0][comb_mask_21],
        overlap_kwargs["p2D21"][:, 1][comb_mask_21],
        c="r",
        s=0.005,
        alpha=0.3,
    )
    axs[1, 0].scatter(
        overlap_kwargs["p2D21"][:, 0][~comb_mask_21],
        overlap_kwargs["p2D21"][:, 1][~comb_mask_21],
        c="b",
        s=0.005,
        alpha=0.3,
    )
    axs[1, 1].imshow(rgb2)
    axs[1, 1].scatter(
        overlap_kwargs["p2D12"][:, 0][comb_mask_12],
        overlap_kwargs["p2D12"][:, 1][comb_mask_12],
        c="r",
        s=0.005,
        alpha=0.3,
    )
    axs[1, 1].scatter(
        overlap_kwargs["p2D12"][:, 0][~comb_mask_12],
        overlap_kwargs["p2D12"][:, 1][~comb_mask_12],
        c="b",
        s=0.005,
        alpha=0.3,
    )

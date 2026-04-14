import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from matplotlib_venn import venn2


def build_patch(
    left_x,
    right_x,
    l_top,
    r_top,
    l_bot,
    r_bot,
    overlap_color=(0.7, 0.35, 0.0, 0.4),
    link_alpha=0.4,
):
    # This is the linkage part
    verts = [
        (left_x, l_top),  # 0  start top-left
        (0.5, l_top),  # 1  curve ctrl1
        (0.5, r_top),  # 2  curve ctrl2
        (right_x, r_top),  # 3  end top-right
        (right_x, r_bot),  # 4  move to bottom-right
        (0.5, r_bot),  # 5  curve ctrl3
        (0.5, l_bot),  # 6  curve ctrl4
        (left_x, l_bot),  # 7  end bottom-left
        (left_x, l_top),  # 8 close
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    patch = PathPatch(
        Path(verts, codes), facecolor=overlap_color, alpha=link_alpha, edgecolor="none"
    )
    return patch


def draw_stackedbars(ax, a, b, bar_colors, bar_width):
    topa, bottoma = a[a > 1].sum(), a[a < 2].sum()
    ax.bar(0, bottoma, width=bar_width, color=bar_colors[0], alpha=0.2)
    ax.bar(0, topa, width=bar_width, color=bar_colors[0], alpha=0.6, bottom=bottoma)
    topb, bottomb = b[b > 1].sum(), b[b < 2].sum()
    ax.bar(1, bottomb, width=bar_width, color=bar_colors[1], alpha=0.2)
    ax.bar(1, topb, width=bar_width, color=bar_colors[1], alpha=0.6, bottom=bottomb)
    return topa, bottoma, topb, bottomb


def get_linkdata(df):
    freq = df.value_counts()
    solo = freq[freq.index.get_level_values(0) < 2]
    solo2 = solo[solo.index.get_level_values(1) < 2].sum()
    solexp = solo[solo.index.get_level_values(1) > 1]
    solexp = np.sum(solexp.index.get_level_values(1) * solexp.values)

    exp = freq[freq.index.get_level_values(0) > 1]
    expsolo = exp[exp.index.get_level_values(1) < 2].sum()
    exp2 = exp[exp.index.get_level_values(1) > 1]
    exp2 = np.sum(exp2.index.get_level_values(1) * exp2.values)
    return solo, solo2, solexp, exp, exp2, expsolo


def add_linkage(ax, df, bottoma, bottomb, bar_width=0.4):
    left_x, right_x = 0 + bar_width / 2, 1 - bar_width / 2
    solo, solo2, solexp, exp, exp2, expsolo = get_linkdata(df)
    patch = build_patch(
        left_x, right_x, solo2, solo2, 0, 0
    )  # cd4 singles to cd8 singles
    ax.add_patch(patch)
    patch = build_patch(
        left_x, right_x, solo.sum(), bottomb + solexp, solo2, bottomb
    )  # cd4 singles to cd8 exp
    ax.add_patch(patch)
    patch = build_patch(
        left_x, right_x, bottoma + expsolo, solo2 + expsolo, bottoma, solo2
    )  # cd4 exp to cd8 singles
    ax.add_patch(patch)
    patch = build_patch(
        left_x,
        right_x,
        bottoma + exp.sum(),
        bottomb + solexp + exp2,
        bottoma + expsolo,
        bottomb + solexp,
    )  # cd4 exp to cd8 exp
    ax.add_patch(patch)


def bar_with_overlap_curved(
    clone4,
    clone8,
    df,
    title_name="title",
    bar_width=0.4,
    bar_colors=("red", "green"),
    overlap_color=(0.7, 0.35, 0.0, 0.4),
):
    """
    Draw two bars (heights a, b) with a curved filled overlap link.
    Overlap height on left bar = x, on right bar = y.
    """
    fig, ax = plt.subplots(figsize=(8, 5), ncols=2, nrows=1)

    # Draw bars
    # pdb.set_trace()
    topa, bottoma, topb, bottomb = draw_stackedbars(
        ax[1], df.cd4_clones, df.cd8_clones, bar_colors, bar_width
    )

    add_linkage(ax[1], df, bottoma, bottomb)
    # Styling
    legend_elements = [
        Patch(facecolor="grey", alpha=0.6, label="Expanded clones"),
        Patch(facecolor="grey", alpha=0.2, label="Single clones"),
    ]
    ax[1].legend(handles=legend_elements, loc="upper center")
    ax[1].set_xlim(-0.5, 1.5)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["CD4 clones", "CD8 clones"])
    ax[1].set_ylabel("Frequency")
    # ax[0].set_title(title_name)

    _ = venn2(
        (set(clone4.keys()), set(clone8.keys())), set_labels=["CD4+", "CD8+"], ax=ax[0]
    )
    # for patch in v.patches:
    #     if patch is not None:
    #         patch.set_transform(Affine2D().rotate_deg_around(0, 0, 90)+ ax[0].transData  # rotate around center
    #         )

    # for txt in v.set_labels + v.subset_labels:
    #     if txt is not None:
    #         txt.set_transform(Affine2D().rotate_deg_around(0, 0, 90)+ax[0].transData)
    ax[0].set_title("unique clones")
    ax[1].set_title("Clone Repertoire")
    fig.suptitle(title_name)
    plt.savefig(f"{title_name}.png", dpi=300)
    plt.close()

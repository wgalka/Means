"""Generate 3D surfaces for every public A_* aggregation class."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggregationslib.aggregations import (
    A_amn, A_ar, A_amx, A_ex, A_ex2, A_ex3, A_gm, A_hm, A_lm, A_lo,
    A_md, A_mn, A_mx, A_ol, A_oln, A_pr, A_pw, A_qd,
)


def aggregation_catalog():
    """Return name, function and a meaningful plotting domain for each class."""
    return [
        ("A_mn: minimum", A_mn(), -2, 2),
        ("A_mx: maximum", A_mx(), -2, 2),
        ("A_pr: product", A_pr(), -2, 2),
        ("A_ar: arithmetic", A_ar(), -2, 2),
        ("A_qd: quadratic", A_qd(), -2, 2),
        ("A_gm: geometric", A_gm(), 0.05, 4),
        ("A_hm: harmonic", A_hm(), 0.05, 4),
        ("A_pw(r=2): power", A_pw(r=2), -2, 2),
        ("A_ex(r=1): exponential", A_ex(r=1), -2, 2),
        ("A_ex2(p=2,q=2)", A_ex2(p=2, q=2), -2, 2),
        ("A_ex3(p=2,q=1.5)", A_ex3(p=2, q=1.5), -2, 2),
        ("A_lm(r=2): Lehmer", A_lm(r=2), 0.05, 4),
        ("A_amn(p=0.5)", A_amn(p=0.5), -2, 2),
        ("A_amx(p=0.5)", A_amx(p=0.5), -2, 2),
        ("A_md: median", A_md(), -2, 2),
        ("A_ol: Olympic", A_ol(), -2, 2),
        ("A_oln(p=1)", A_oln(p=1), -2, 2),
        ("A_lo: logarithmic", A_lo(), 0.05, 4),
    ]


def evaluate_surface(function, lower, upper, samples=61):
    """Evaluate a binary aggregation on a square grid, masking invalid points."""
    values = np.linspace(lower, upper, samples)
    x, y = np.meshgrid(values, values)
    z = np.full_like(x, np.nan, dtype=float)

    for row in range(samples):
        for column in range(samples):
            try:
                result = function(np.array([x[row, column], y[row, column]]))
                value = float(np.real(result))
                if np.isfinite(value):
                    z[row, column] = value
            except (ArithmeticError, ValueError, ZeroDivisionError, FloatingPointError):
                pass

    return x, y, z


def plot_surface(axis, title, function, lower, upper):
    x, y, z = evaluate_surface(function, lower, upper)
    axis.plot_surface(x, y, z, cmap="viridis", linewidth=0, antialiased=True)
    axis.set_title(f"{title}\nrange: [{lower:g}, {upper:g}]", fontsize=9)
    axis.set_xlabel("x", labelpad=2)
    axis.set_ylabel("y", labelpad=2)
    axis.set_zlabel("A(x,y)", labelpad=2)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    finite_values = z[np.isfinite(z)]
    if finite_values.size:
        axis.set_zlim(np.nanmin(finite_values), np.nanmax(finite_values))
    axis.view_init(elev=28, azim=-125)


def main():
    output_dir = Path(__file__).resolve().parent / "docs" / "assets" / "aggregations"
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = aggregation_catalog()

    columns = 3
    rows = (len(catalog) + columns - 1) // columns
    figure = plt.figure(figsize=(18, rows * 5.5))
    for index, (title, function, lower, upper) in enumerate(catalog, start=1):
        axis = figure.add_subplot(rows, columns, index, projection="3d")
        plot_surface(axis, title, function, lower, upper)

    figure.suptitle(
        "All aggregations in aggregationslib on their natural domains",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    overview = output_dir.parent / "aggregation-surfaces-all.png"
    figure.savefig(overview, dpi=160, bbox_inches="tight")
    plt.close(figure)

    for title, function, lower, upper in catalog:
        safe_name = title.split(":", 1)[0].replace("(", "_").replace(")", "")
        single = plt.figure(figsize=(8, 7))
        axis = single.add_subplot(111, projection="3d")
        plot_surface(axis, title, function, lower, upper)
        single.tight_layout()
        single.savefig(output_dir / f"{safe_name}.png", dpi=180, bbox_inches="tight")
        plt.close(single)

    print(f"Generated overview: {overview}")
    print(f"Generated individual plots: {len(catalog)}")


if __name__ == "__main__":
    main()

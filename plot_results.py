"""
plot_results.py — Exemplos de ajuste de curvas de Bézier para o relatório
Execute: python plot_results.py
Gera: fig1_arco.pdf, fig2_tolerancias.pdf, fig3_contorno.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from fitCurves import fit_curve
from bezier import q

# Estilo geral
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_result(points, max_error, ax, title):
    """Plota pontos de entrada e curva(s) ajustada(s) em um eixo."""
    curves = fit_curve(points, max_error)
    t = np.linspace(0, 1, 200)

    # Polilinha de entrada
    ax.plot(*points.T, color="lightgray", linewidth=1,
            linestyle="--", zorder=1)

    # Curvas ajustadas
    for i, curve in enumerate(curves):
        pts = np.array([q(curve, ti) for ti in t])
        ax.plot(*pts.T, color="steelblue", linewidth=2,
                label="Bézier" if i == 0 else "")
        # Pontos de controle internos
        ax.plot(*curve[1], "o", color="orange", markersize=5, zorder=4)
        ax.plot(*curve[2], "o", color="orange", markersize=5, zorder=4)

    # Pontos de entrada
    ax.scatter(*points.T, color="crimson", zorder=5, s=40,
               label="Pontos")

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Anotação: número de segmentos
    ax.text(0.02, 0.02, f"{len(curves)} segmento(s)",
            transform=ax.transAxes, fontsize=9, color="gray")


# Figura 1 — Arco suave com poucos pontos
points_arco = np.array([
    [0.0, 0.0],
    [1.0, 1.5],
    [2.5, 2.2],
    [4.0, 1.8],
    [5.0, 0.5],
    [5.5, -0.5],
], dtype=float)

fig, ax = plt.subplots(figsize=(5, 3.5))
plot_result(points_arco, max_error=10, ax=ax,
            title="Arco suave — 6 pontos, max_error = 10")
handles = [
    plt.Line2D([0], [0], color="steelblue", linewidth=2, label="Curva ajustada"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
               markersize=7, label="Pontos de entrada"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange",
               markersize=7, label="Pontos de controle"),
]
ax.legend(handles=handles, loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("fig1_arco.pdf", bbox_inches="tight")
plt.savefig("fig1_arco.png", dpi=150, bbox_inches="tight")
print("Salvo: fig1_arco.pdf / .png")


# Figura 2 — Comparação de tolerâncias (mesmo conjunto de pontos)
points_s = np.array([
    [0.0,  0.0],
    [0.5,  1.0],
    [1.5,  2.0],
    [2.5,  2.5],
    [3.5,  2.0],
    [4.5,  1.0],
    [5.5,  0.5],
    [6.5,  1.5],
    [7.5,  2.5],
    [8.0,  2.0],
], dtype=float)

erros = [5, 20, 80]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, err in zip(axes, erros):
    plot_result(points_s, max_error=err, ax=ax,
                title=f"max_error = {err}")
fig.suptitle("Efeito da tolerância sobre o número de segmentos",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig("fig2_tolerancias.pdf", bbox_inches="tight")
plt.savefig("fig2_tolerancias.png", dpi=150, bbox_inches="tight")
print("Salvo: fig2_tolerancias.pdf / .png")


# Figura 3 — Contorno irregular com subdivisão recursiva
points_contorno = np.array([
    [0.0,  0.0],
    [0.8,  1.8],
    [1.0,  0.5],
    [2.0,  2.5],
    [2.5,  0.2],
    [3.5,  2.8],
    [4.0,  0.8],
    [5.0,  2.0],
    [5.8,  0.3],
    [6.5,  1.5],
    [7.0,  0.0],
], dtype=float)

fig, ax = plt.subplots(figsize=(6, 3.5))
plot_result(points_contorno, max_error=10, ax=ax,
            title="Contorno irregular — 11 pontos, max_error = 10")
plt.tight_layout()
plt.savefig("fig3_contorno.pdf", bbox_inches="tight")
plt.savefig("fig3_contorno.png", dpi=150, bbox_inches="tight")
print("Salvo: fig3_contorno.pdf / .png")

plt.show()
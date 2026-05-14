"""
Avaliação de curvas de Bézier cúbicas
Implementação Python 3 modernizada do algoritmo de Schneider (1990).

Referências:
    Schneider, P. J. (1990). An Algorithm for Automatically Fitting Digitized Curves.
    In A. Glassner (Ed.), Graphics Gems. Academic Press.

    Implementação original (Python 2):
    Volker Poplawski — https://github.com/volkerp/fitCurves
"""

import numpy as np


def q(ctrl_poly: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Avalia a curva de Bézier cúbica no(s) parâmetro(s) t.

    Args:
        ctrl_poly: Array (4, 2) com os pontos de controle [P0, P1, P2, P3].
        t:         Escalar ou array de parâmetros em [0, 1].

    Returns:
        Ponto(s) na curva como array NumPy.
    """
    ctrl_poly = np.asarray(ctrl_poly, dtype=float)
    t = np.asarray(t, dtype=float)
    t_ = 1.0 - t
    return (
        t_**3              * ctrl_poly[0]
        + 3 * t_**2 * t    * ctrl_poly[1]
        + 3 * t_    * t**2 * ctrl_poly[2]
        +             t**3 * ctrl_poly[3]
    )


def qprime(ctrl_poly: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Avalia a primeira derivada da curva de Bézier cúbica em t.

    Args:
        ctrl_poly: Array (4, 2) com os pontos de controle.
        t:         Escalar ou array de parâmetros em [0, 1].

    Returns:
        Vetor(es) tangente(s) na curva.
    """
    ctrl_poly = np.asarray(ctrl_poly, dtype=float)
    t = np.asarray(t, dtype=float)
    t_ = 1.0 - t
    return (
        3 * t_**2          * (ctrl_poly[1] - ctrl_poly[0])
        + 6 * t_ * t       * (ctrl_poly[2] - ctrl_poly[1])
        + 3 * t**2         * (ctrl_poly[3] - ctrl_poly[2])
    )


def qprimeprime(ctrl_poly: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Avalia a segunda derivada da curva de Bézier cúbica em t.

    Args:
        ctrl_poly: Array (4, 2) com os pontos de controle.
        t:         Escalar ou array de parâmetros em [0, 1].

    Returns:
        Vetor(es) de curvatura na curva.
    """
    ctrl_poly = np.asarray(ctrl_poly, dtype=float)
    t = np.asarray(t, dtype=float)
    t_ = 1.0 - t
    return (
        6 * t_ * (ctrl_poly[2] - 2 * ctrl_poly[1] + ctrl_poly[0])
        + 6 * t  * (ctrl_poly[3] - 2 * ctrl_poly[2] + ctrl_poly[1])
    )

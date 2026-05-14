"""
Ajuste automático de curvas de Bézier cúbicas a dados discretos
Implementação Python 3 modernizada do algoritmo de Schneider (1990).

Referências:
    Schneider, P. J. (1990). An Algorithm for Automatically Fitting Digitized Curves.
    In A. Glassner (Ed.), Graphics Gems. Academic Press.

    Implementação original (Python 2):
    Volker Poplawski — https://github.com/volkerp/fitCurves
"""

from __future__ import annotations

import numpy as np
from bezier import q, qprime, qprimeprime


# Tipo de dado para uma curva de Bézier cúbica: lista de 4 pontos de controle (P0, P1, P2, P3)
BezierCurve = list[np.ndarray]


# Função principal: ajuste de uma ou mais curvas de Bézier a um conjunto de pontos
def fit_curve(points: np.ndarray, max_error: float) -> list[BezierCurve]:
    """Ajusta uma ou mais curvas de Bézier cúbicas a um conjunto de pontos.

    Args:
        points:    Array (N, 2) de pontos 2-D ordenados.
        max_error: Tolerância máxima de erro (distância ao quadrado).

    Returns:
        Lista de curvas de Bézier, cada uma representada por 4 pontos de controle.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("São necessários pelo menos 2 pontos.")

    # Calcula vetores unitários de direção nas extremidades baseado na inclinação entre os primeiros e últimos dois pontos
    left_tangent  = _normalize(points[1]  - points[0])
    right_tangent = _normalize(points[-2] - points[-1])
    
    # Inicia o processo recursivo de ajuste de Schneider
    return _fit_cubic(points, left_tangent, right_tangent, max_error)


# Parte recursiva do ajuste: tenta ajustar uma curva, verifica o erro e divide se necessário
def _fit_cubic(points: np.ndarray, left_tangent: np.ndarray, right_tangent: np.ndarray, error: float) -> list[BezierCurve]:
    """Ajusta recursivamente curvas cúbicas a um segmento de pontos."""

    # Caso base: quando há apenas 2 pontos a é heurística simples
    if len(points) == 2:
        dist = np.linalg.norm(points[0] - points[1]) / 3.0
        bez = [
            points[0],
            points[0] + left_tangent  * dist,
            points[1] + right_tangent * dist,
            points[1],
        ]
        return [bez]

    # Parametrização inicial por comprimento de corda
    u = _chord_length_parameterize(points)
    bez = _generate_bezier(points, u, left_tangent, right_tangent)

    # Checa erro máximo
    max_err, split_point = _compute_max_error(points, bez, u)
    if max_err < error:
        return [bez]

    # Se o erro for tolerável, tenta reparametrização iterativa (Newton-Raphson)
    if max_err < error ** 2:
        for _ in range(20):
            u_prime = _reparameterize(bez, points, u)
            bez      = _generate_bezier(points, u_prime, left_tangent, right_tangent)
            max_err, split_point = _compute_max_error(points, bez, u_prime)
            if max_err < error:
                return [bez]
            u = u_prime

    # Se o ajuste falha, então divide no ponto de maior erro e chama a função recursivamente
    center_tangent = _normalize(points[split_point - 1] - points[split_point + 1])
    return (
        _fit_cubic(points[: split_point + 1],  left_tangent,   center_tangent, error)
      + _fit_cubic(points[split_point:],     -center_tangent, right_tangent,  error)
    )

# Geração da curva com mínimos quadrados
def _generate_bezier(points: np.ndarray, parameters: list[float], left_tangent: np.ndarray, right_tangent: np.ndarray) -> BezierCurve:
    """Calcula os pontos de controle internos via mínimos quadrados (método de Schneider)."""

    u = np.asarray(parameters) # (N,)
    t_ = 1.0 - u # (N,)

    # Matriz A (N, 2, 2) onde A[i][0] é o vetor associado a left_tangent e A[i][1] ao right_tangent para o parâmetro u[i]
    # Usada por Schneider para auxiliar os cálculos
    A = np.zeros((len(u), 2, 2))
    A[:, 0] = (left_tangent  * (3 * t_**2 * u)    [:, np.newaxis])
    A[:, 1] = (right_tangent * (3 * t_    * u**2)  [:, np.newaxis])

    # Montagem da matriz C usando cálculo otimizado em tesnores do numpy (evitando loops explícitos como o da implementação original)
    C = np.einsum("ijk,ijl->kl", A, A)          # C[k,l] = sum_i dot(A[i,k], A[i,l])

    # Termo independente: ponto observado menos Bézier degenerada (P0..P0..P3..P3)
    bez_degenerate = np.array([points[0], points[0], points[-1], points[-1]])
    tmp = points - np.array([q(bez_degenerate, ui) for ui in u])  # (N, 2)

    # Montagem da matriz X usando cálculo otimizado em tesnores do numpy
    X = np.einsum("ijk,ij->k", A, tmp)          # X[k] = sum_i dot(A[i,k], tmp[i])

    # Determinantes de C e X
    det_C0_C1 = C[0, 0] * C[1, 1] - C[1, 0] * C[0, 1]
    det_X_C1  = X[0]    * C[1, 1] - X[1]    * C[0, 1]
    det_C0_X  = C[0, 0] * X[1]    - C[1, 0] * X[0]

    # acha os valores dos alfas
    alpha_l = 0.0 if det_C0_C1 == 0.0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0.0 else det_C0_X / det_C0_C1

    # Heurística Wu/Barsky para alfas degenerados que é usada na implementação original
    seg_length = np.linalg.norm(points[0] - points[-1])
    epsilon    = 1.0e-6 * seg_length

    if alpha_l < epsilon or alpha_r < epsilon:
        d = seg_length / 3.0
        p1 = points[0]  + left_tangent  * d
        p2 = points[-1] + right_tangent * d
    else:
        p1 = points[0]  + left_tangent  * alpha_l
        p2 = points[-1] + right_tangent * alpha_r

    return [points[0], p1, p2, points[-1]]


# Reparametrização com método de Newton-Raphson
def _reparameterize(bez: BezierCurve, points: np.ndarray, parameters: list[float]) -> list[float]:
    return [_newton_raphson_root_find(bez, pt, u) for pt, u in zip(points, parameters)]


def _newton_raphson_root_find(bez: BezierCurve, point: np.ndarray, u: float) -> float:
    """Refinamento do parâmetro u minimizando a distância ponto–curva."""
    d          = q(bez, u) - point
    d_prime    = qprime(bez, u)
    d_prime2   = qprimeprime(bez, u)

    numerator   = float(np.dot(d, d_prime))
    denominator = float(np.dot(d_prime, d_prime) + np.dot(d, d_prime2))

    if denominator == 0.0:
        return u
    return u - numerator / denominator


# Parametrização com comprimento de corda
def _chord_length_parameterize(points: np.ndarray) -> list[float]:
    """Parametrização proporcional ao comprimento acumulado das cordas."""
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    u     = np.concatenate([[0.0], np.cumsum(diffs)])
    total = u[-1]
    if total == 0.0:
        raise ValueError("Pontos coincidentes; impossível parametrizar.")
    return (u / total).tolist()

# Cálculo do erro máximo
def _compute_max_error(points: np.ndarray, bez: BezierCurve, parameters: list[float]) -> tuple[float, int]:
    """Retorna (erro_máximo_quadrático, índice_do_ponto_mais_distante)."""
    u      = np.asarray(parameters)
    fitted = np.array([q(bez, ui) for ui in u])   # (N, 2)
    dists  = np.sum((fitted - points) ** 2, axis=1)

    idx = int(np.argmax(dists))
    # Garante que o ponto de divisão não seja uma extremidade
    if idx == 0:
        idx = 1
    if idx >= len(points) - 1:
        idx = len(points) - 2

    return float(dists[idx]), idx


# Retorna o unitário de um vetor dado
def _normalize(v: np.ndarray) -> np.ndarray:
    """Retorna o vetor unitário de v. Levanta ValueError se v for nulo."""
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError(f"Tentativa de normalizar vetor nulo: {v}")
    return v / norm

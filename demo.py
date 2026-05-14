"""
Interface gráfica Tkinter para ajuste de curvas de Bézier
Modernizado para Python 3.

Uso:
    python demo.py

Controles:
    Botão esquerdo   — adiciona ponto / arrasta ponto existente
    Botão direito    — remove o último ponto
    Scroll "Max Error" — ajusta a tolerância de ajuste

Implementação original (Python 2):
Volker Poplawski — https://github.com/volkerp/fitCurves
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

import bezier as bz
from fitCurves import fit_curve


# Utilidades de geometria e canvas
def _center_of_bbox(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    return x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2


class BezierCanvas(tk.Canvas):
    """Canvas com métodos auxiliares para desenhar curvas de Bézier."""

    BEZIER_STEPS = 50

    def create_bezier(self, ctrl: list[np.ndarray], tag: str) -> None:
        """Desenha a curva e os segmentos dos pontos de controle."""
        t_vals  = np.linspace(0.0, 1.0, self.BEZIER_STEPS + 1)
        curve   = [bz.q(ctrl, t).tolist() for t in t_vals]

        # Linha da curva
        for p1, p2 in zip(curve, curve[1:]):
            self.create_line(*p1, *p2, fill="#1a73e8", width=2, tag=tag)

        # Segmentos de controle e marcadores
        self._draw_control_segment(ctrl[0], ctrl[1], tag)
        self._draw_control_segment(ctrl[3], ctrl[2], tag)

    def _draw_control_segment(
        self, anchor: np.ndarray, handle: np.ndarray, tag: str
    ) -> None:
        self.create_line(*anchor.tolist(), *handle.tolist(),
                         fill="#aaa", dash=(4, 2), tag=tag)
        self._create_point(*handle.tolist(), radius=3,
                           fill="#333", outline="#fff", tag=tag)

    def _create_point(
        self, x: float, y: float, radius: int = 4, **kwargs
    ) -> int:
        return self.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            **kwargs,
        )

    def create_input_point(self, x: float, y: float) -> int:
        """Cria o marcador vermelho de um ponto de entrada."""
        return self._create_point(x, y, radius=5,
                                  fill="#e53935", outline="#b71c1c",
                                  tags="point")

    def point_center(self, item_id: int) -> tuple[float, float]:
        return _center_of_bbox(*self.coords(item_id))

    def items_at(self, x: float, y: float, tag: str) -> list[int]:
        return [
            item for item in self.find_overlapping(x, y, x, y)
            if tag in self.gettags(item)
        ]


# Classe Principal
class App:
    """Janela principal da demo de ajuste de Bézier."""

    CANVAS_W = 700
    CANVAS_H = 500
    DEFAULT_ERROR = 10.0

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Ajuste de Curvas de Bézier — Schneider (1990)")
        self.root.resizable(False, False)

        self._point_ids: list[int] = []
        self._dragging: int | None = None

        self._build_ui()
        self._bind_events()

    # Fazendo a UI
    def _build_ui(self) -> None:
        # Canvas
        self.canvas = BezierCanvas(
            self.root, bg="#f8f9fa",
            width=self.CANVAS_W, height=self.CANVAS_H,
            cursor="crosshair",
        )
        self.canvas.pack(side=tk.LEFT, padx=4, pady=4)

        # Painel lateral
        sidebar = ttk.Frame(self.root, padding=12)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)

        ttk.Label(sidebar, text="Parâmetros", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, pady=6)

        ttk.Label(sidebar, text="Max Error (tolerância)").pack(anchor="w")
        self._error_var = tk.DoubleVar(value=self.DEFAULT_ERROR)
        error_spin = ttk.Spinbox(
            sidebar, from_=0.1, to=1_000_000.0, increment=1.0,
            textvariable=self._error_var, width=10,
            command=self._redraw,
        )
        error_spin.pack(anchor="w", pady=(2, 12))

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, pady=6)
        ttk.Label(sidebar, text="Ajuda", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Label(sidebar, text=(
            "Clique esquerdo — adiciona ponto\n"
            "Arraste        — move ponto\n"
            "Clique direito — remove último ponto\n"
            "Spinbox        — ajusta tolerância"
        ), justify="left").pack(anchor="w", pady=4)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, pady=6)
        ttk.Button(sidebar, text="Limpar tudo", command=self._clear).pack(fill=tk.X)

        # Barra de status
        self._status = tk.StringVar(value="Adicione pontos para começar.")
        ttk.Label(self.root, textvariable=self._status, relief=tk.SUNKEN,
                  anchor="w").pack(side=tk.BOTTOM, fill=tk.X, ipady=2)

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>",   self._on_left_press)
        self.canvas.bind("<B1-Motion>",       self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-3>",   self._on_right_press)   # botão direito
        self.canvas.bind("<ButtonPress-2>",   self._on_right_press)   # meio (compatibilidade)

    # Mouse
    def _on_left_press(self, event: tk.Event) -> None:
        hits = self.canvas.items_at(event.x, event.y, "point")
        if hits:
            self._dragging = hits[0]
        else:
            pid = self.canvas.create_input_point(event.x, event.y)
            self._point_ids.append(pid)
            self._redraw()

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self._dragging is not None:
            r = 5
            self.canvas.coords(self._dragging,
                                event.x - r, event.y - r,
                                event.x + r, event.y + r)
            self._redraw()

    def _on_left_release(self, _event: tk.Event) -> None:
        self._dragging = None

    def _on_right_press(self, _event: tk.Event) -> None:
        if self._point_ids:
            self.canvas.delete(self._point_ids.pop())
            self._redraw()

   # Redesenha a curva ajustada
    def _redraw(self) -> None:
        # Redesenha a polilinha de entrada
        self.canvas.delete("polyline")
        centers = [self.canvas.point_center(pid) for pid in self._point_ids]
        if len(centers) >= 2:
            for p1, p2 in zip(centers, centers[1:]):
                self.canvas.create_line(*p1, *p2,
                                        fill="#ccc", dash=(3, 3),
                                        tag="polyline")
            self.canvas.tag_lower("polyline")

        # Redesenha as curvas de Bézier ajustadas
        self.canvas.delete("bezier")
        if len(self._point_ids) < 2:
            self._status.set(f"Pontos: {len(self._point_ids)}  — adicione pelo menos 2.")
            return

        pts = np.array(centers)
        try:
            error = float(self._error_var.get()) ** 2
            curves = fit_curve(pts, error)
            for crv in curves:
                self.canvas.create_bezier(crv, tag="bezier")
            self._status.set(
                f"Pontos: {len(pts)}  |  Segmentos Bézier: {len(curves)}  "
                f"|  Max Error²: {error:.1f}"
            )
        except Exception as exc:
            self._status.set(f"Erro no ajuste: {exc}")

    def _clear(self) -> None:
        for pid in self._point_ids:
            self.canvas.delete(pid)
        self._point_ids.clear()
        self.canvas.delete("polyline")
        self.canvas.delete("bezier")
        self._status.set("Canvas limpo. Adicione pontos para começar.")

    # Execução
    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    App().run()

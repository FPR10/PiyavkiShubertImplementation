"""
Modulo di visualizzazione per l'algoritmo di Piyavski-Shubert.

Funzioni pubbliche:
    plot_result(tf, result)      — pannello statico completo (3 grafici)
    step_visualizer(tf, result)  — navigatore passo-passo interattivo

Entrambe accettano un TestFunction e un PSResult calcolato con
store_candidates=True.  Tutte le dipendenze da matplotlib sono locali
a questo modulo.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from dataclasses import dataclass
from typing import Callable


# ── Palette ────────────────────────────────────────────────────────────────
_C_FUNC  = "#2563EB"   # blu      – curva della funzione
_C_TENT  = "#F59E0B"   # arancio  – tende di Lipschitz
_C_OPT   = "#DC2626"   # rosso    – punto ottimale
_C_NEW   = "#7C3AED"   # viola    – ultimo punto valutato (step-mode)
_C_CONV  = "#16A34A"   # verde    – curva di convergenza
_C_BG    = "#F8FAFC"   # sfondo assi

# Margine verticale (frazione del range della funzione) usato da _fix_axes
_Y_PAD_FRAC = 0.15


# ── TestFunction ───────────────────────────────────────────────────────────

@dataclass
class TestFunction:
    """
    Wrapper leggero che accoppia una funzione al suo dominio e alla
    costante di Lipschitz L.

    Attributi
    ---------
    name : etichetta (es. "f2").
    func : callable f(x) -> float.
    a, b : estremi dell'intervallo.
    L    : costante di Lipschitz.
    """
    name : str
    func : Callable[[float], float]
    a    : float
    b    : float
    L    : float

    def __call__(self, x: float) -> float:
        return self.func(x)


# ── Helpers interni ────────────────────────────────────────────────────────

def _sample(tf: TestFunction, n: int = 800):
    """Campiona uniformemente la funzione su [a, b]."""
    xs = np.linspace(tf.a, tf.b, n)
    ys = np.array([tf(x) for x in xs])
    return xs, ys


def _y_limits(ys: np.ndarray, pad: float = _Y_PAD_FRAC):
    """Restituisce (y_min, y_max) con margine relativo al range."""
    y_range = ys.max() - ys.min()
    margin  = pad * y_range if y_range > 0 else 0.1
    return ys.min() - margin, ys.max() + margin


def _fix_axes(ax, xs: np.ndarray, ys: np.ndarray, x_pad_frac=0.1):
    x_range = xs[-1] - xs[0]
    x_margin = x_pad_frac * x_range

    ax.set_xlim(xs[0] - x_margin, xs[-1] + x_margin)
    ax.set_ylim(*_y_limits(ys))


def _style_ax(ax):
    ax.set_facecolor(_C_BG)
    ax.grid(True, linestyle="--", alpha=0.35)


def _draw_function(ax, xs, ys, tf_name: str):
    """Disegna la curva della funzione con area riempita."""
    ax.plot(xs, ys, color=_C_FUNC, linewidth=1.8, label=f"{tf_name}(x)", zorder=3)
    ax.fill_between(xs, ys, ys.min() - 0.05 * (ys.max() - ys.min()),
                    alpha=0.07, color=_C_FUNC)


def _draw_tents(ax, candidates, tf_L: float):
    """
    Disegna le tende di Lipschitz per tutti i candidati.

    Non tocca i limiti degli assi: si usa _fix_axes() dopo la chiamata
    per congelare il viewport sulla funzione campionata.
    """
    for cand in candidates:
        xl, xr  = cand.x_left, cand.x_right
        x_tent  = np.linspace(xl, xr, 60)
        y_left  = cand.func_left  - tf_L * (x_tent - xl)
        y_right = cand.func_right + tf_L * (x_tent - xr)
        ax.plot(x_tent, np.maximum(y_left, y_right),
                color=_C_TENT, linewidth=0.9, alpha=0.55, zorder=2)


def _draw_optimum(ax, x_opt: float, f_opt: float):
    ax.scatter([x_opt], [f_opt], color=_C_OPT, s=100, zorder=7, marker="*",
               label=f"x*={x_opt:.4f}  f*={f_opt:.4f}")


def _draw_convergence(ax, iters, fvals, f_opt: float):
    ax.plot(iters, fvals, color=_C_CONV, linewidth=1.8,
            marker="o", markersize=3.5, label="f*(k)")
    ax.axhline(f_opt, color=_C_OPT, linestyle="--",
               linewidth=1.1, label=f"f* finale: {f_opt:.4f}")
    ax.set_xlabel("Iterazione")
    ax.set_ylabel("f*(k)")
    ax.legend(fontsize=8)


# ══════════════════════════════════════════════════════════════════════════
#  1. PLOTTING STATICO
# ══════════════════════════════════════════════════════════════════════════

def plot_result(tf: TestFunction, result) -> plt.Figure:
    """
    Pannello statico con tre grafici:
      • in alto a sinistra : funzione e punto ottimo trovato
      • in alto a destra   : funzione con tutti i punti valutati
      • in basso (intero)  : curva di convergenza f*(k)

    Parametri
    ---------
    tf     : TestFunction
    result : PSResult
    """
    xs, ys = _sample(tf)

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(_C_BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_func = fig.add_subplot(gs[0, 0])
    ax_pts  = fig.add_subplot(gs[0, 1])
    ax_conv = fig.add_subplot(gs[1, :])

    # ── Funzione + ottimo ──────────────────────────────────────────────────
    _style_ax(ax_func)
    _draw_function(ax_func, xs, ys, tf.name)
    _draw_optimum(ax_func, result.x_opt, result.f_opt)
    #_fix_axes(ax_func, xs, ys)
    ax_func.set_xlabel("x")
    ax_func.set_ylabel("f(x)")
    ax_func.set_title(f"{tf.name} — ottimo trovato", fontsize=10)
    ax_func.legend(fontsize=8)

    # ── Punti valutati ─────────────────────────────────────────────────────
    _style_ax(ax_pts)
    _draw_function(ax_pts, xs, ys, tf.name)
    eval_xs = [h[1] for h in result.history]
    eval_ys = [tf(x) for x in eval_xs]
    ax_pts.scatter(eval_xs, eval_ys, color=_C_OPT, s=30, zorder=5,
                   label=f"Punti valutati ({len(eval_xs)})")
    _draw_optimum(ax_pts, result.x_opt, result.f_opt)
    #_fix_axes(ax_pts, xs, ys)
    ax_pts.set_xlabel("x")
    ax_pts.set_ylabel("f(x)")
    ax_pts.set_title(f"{tf.name} — punti valutati", fontsize=10)
    ax_pts.legend(fontsize=8)

    # ── Convergenza ────────────────────────────────────────────────────────
    _style_ax(ax_conv)
    iters = [h[0] for h in result.history]
    fvals = [h[2] for h in result.history]
    _draw_convergence(ax_conv, iters, fvals, result.f_opt)
    ax_conv.set_title("Convergenza dell'algoritmo", fontsize=10)

    fig.suptitle(
        f"Piyavski-Shubert — {tf.name}   "
        f"iterazioni: {result.iterations}   valutazioni: {result.n_evals}",
        fontsize=13, fontweight="bold",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  2. STEP VISUALIZER
# ══════════════════════════════════════════════════════════════════════════

class _StepVisualizer:
    """
    Navigatore passo-passo.

    Riceve un PSResult già calcolato con store_candidates=True e naviga
    tra gli snapshot della heap con i pulsanti Prev / Next / Reset.
    Non riesegue mai l'algoritmo: è una pura vista sui dati.
    """

    def __init__(self, tf: TestFunction, result):
        if not result.candidates_log:
            raise ValueError(
                "PSResult.candidates_log è vuoto: ricalcola con store_candidates=True."
            )

        self._tf     = tf
        self._result = result
        self._n      = len(result.candidates_log)
        self._idx    = 0

        self._xs, self._ys = _sample(tf)
        self._build_figure()
        self._draw()

    # ── Layout ─────────────────────────────────────────────────────────────

    def _build_figure(self):
        self._fig = plt.figure(figsize=(13, 6))
        self._fig.patch.set_facecolor(_C_BG)

        gs = gridspec.GridSpec(
            2, 2, figure=self._fig,
            height_ratios=[10, 1], hspace=0.45, wspace=0.35,
        )
        self._ax_main = self._fig.add_subplot(gs[0, 0])
        self._ax_conv = self._fig.add_subplot(gs[0, 1])

        btn_gs = gridspec.GridSpecFromSubplotSpec(
            1, 7, subplot_spec=gs[1, :], wspace=0.2,
        )
        ax_prev  = self._fig.add_subplot(btn_gs[0, 1])
        ax_label = self._fig.add_subplot(btn_gs[0, 3])
        ax_next  = self._fig.add_subplot(btn_gs[0, 5])
        ax_reset = self._fig.add_subplot(btn_gs[0, 6])

        self._btn_prev  = Button(ax_prev,  "◀  Prev",  color="#E2E8F0", hovercolor="#CBD5E1")
        self._btn_next  = Button(ax_next,  "Next  ▶",  color="#DBEAFE", hovercolor="#BFDBFE")
        self._btn_reset = Button(ax_reset, "↺ Reset",  color="#FEE2E2", hovercolor="#FECACA")

        ax_label.set_axis_off()
        self._label_ax = ax_label

        self._btn_prev.on_clicked(self._on_prev)
        self._btn_next.on_clicked(self._on_next)
        self._btn_reset.on_clicked(self._on_reset)

    # ── Ridisegno ───────────────────────────────────────────────────────────

    def _draw(self):
        k      = self._idx
        result = self._result
        tf     = self._tf

        snapshot          = result.candidates_log[k]
        iter_k, x_k, f_k = result.history[k]

        # ── Grafico principale ─────────────────────────────────────────────
        ax = self._ax_main
        ax.cla()
        _style_ax(ax)

        _draw_function(ax, self._xs, self._ys, tf.name)
        _draw_tents(ax, snapshot, tf.L)

        if k > 0:
            _, x_prev, _ = result.history[k - 1]
            ax.scatter([x_prev], [tf(x_prev)],
                       color=_C_NEW, s=60, zorder=6,
                       label=f"Ultimo val.: x={x_prev:.4f}")

        _draw_optimum(ax, x_k, f_k)

        # Blocca il viewport DOPO aver aggiunto tende e scatter
        #_fix_axes(ax, self._xs, self._ys)

        status = "✔ Converge" if k == self._n - 1 else f"heap: {len(snapshot)} candidati"
        ax.set_title(
            f"{tf.name}   iter {iter_k}   eval {k+2}   {status}",
            fontsize=10,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend(fontsize=8, loc="upper right")

        # ── Curva di convergenza ───────────────────────────────────────────
        ax2 = self._ax_conv
        ax2.cla()
        _style_ax(ax2)

        iters_so_far = [result.history[i][0] for i in range(k + 1)]
        fvals_so_far = [result.history[i][2] for i in range(k + 1)]
        _draw_convergence(ax2, iters_so_far, fvals_so_far, f_k)
        ax2.set_title("Convergenza", fontsize=10)

        # ── Etichetta centrale ─────────────────────────────────────────────
        self._label_ax.cla()
        self._label_ax.set_axis_off()
        self._label_ax.text(
            0.5, 0.5,
            f"step {k + 1} / {self._n}",
            ha="center", va="center",
            fontsize=10, color="#475569",
            transform=self._label_ax.transAxes,
        )

        self._fig.canvas.draw_idle()

    # ── Callback pulsanti ──────────────────────────────────────────────────

    def _on_next(self, _):
        if self._idx < self._n - 1:
            self._idx += 1
            self._draw()

    def _on_prev(self, _):
        if self._idx > 0:
            self._idx -= 1
            self._draw()

    def _on_reset(self, _):
        self._idx = 0
        self._draw()

    def show(self):
        plt.show()


def step_visualizer(tf: TestFunction, result) -> None:
    """
    Apre la finestra di navigazione passo-passo.

    Parametri
    ---------
    tf     : TestFunction
    result : PSResult calcolato con store_candidates=True
    """
    vis = _StepVisualizer(tf, result)
    vis.show()
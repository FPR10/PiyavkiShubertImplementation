"""
Modalità di avvio dell'algoritmo:

- Da terminale:
    python main.py --fn 5

- Da terminale ridefinendo i parametri:
    python main.py --fn 5 --a 0 --b 1.2 --L 36 --tol 1e-5 --max-iter 5000
 
- Grafico statico completo:
    python main.py --fn 5 --mode plot
 
- Grafico step-by-step:
    python main.py --fn 5 --mode step
 
- run su tutte le 20 funzioni:
     python main.py --allFunctions
 
- Lista delle funzioni disponibili:
    python main.py --list
 
  
Senza argomenti → menu interattivo (avvio tramite IDE).
"""


import sys
import time
import argparse
 
from testFunctions import TEST_REGISTRY
from PiyavskiShubert import piShAlgorithm
from visualizer import TestFunction
 
 
# ── Utils ──────────────────────────────────────────────────────

SEP  = "=" * 100
SSEP = "-" * 100
 
 
#Titolo tabella
def _header():
    print()
    print()
    print(SEP)
    print("   ALGORITMO DI PIYAVSKI-SHUBERT")
    print(SEP)
    print(f"  {'#':>2}  {'[a, b]':^22}  {'L':>9}  {'x*':>12}  "
          f"{'f(x*)':>12}  {'iter':>6}  {'eval':>6}  {'ms':>8}")
    print(SSEP)
 
 
def _row(idx, a, b, L, res, ms):
    print(f"  {idx:>2}  {f'[{a}, {b}]':^22}  {L:>9.2f}  "
          f"{res.x_opt:>12.6f}  {res.f_opt:>12.6f}  "
          f"{res.iterations:>6}  {res.n_evals:>6}  {ms:>7.2f}ms")
 
 
def _detail(res):
    print(f"\n  x*         = {res.x_opt:.10f}")
    print(f"  f(x*)      = {res.f_opt:.10f}")
    print(f"  iterazioni = {res.iterations}")
    print(f"  valutazioni= {res.n_evals}")
    print(SEP)
 
 
# ── Esecuzione singola ─────────────────────────────────────────────────────
 
def run_single(idx: int, func, a: float, b: float, L: float,
               tol: float, max_iter: int, mode: str) -> None:
    """Esegue l'algoritmo e mostra l'output nella modalità richiesta."""
 
    # store_candidates serve sia per plot che per step
    store = (mode == "plot" or mode == "step")
 
    t0 = time.perf_counter()
    result = piShAlgorithm(func, a, b, L,
                              tol=tol, max_iter=max_iter,
                              store_candidates=store)
    ms = (time.perf_counter() - t0) * 1000
 
    _header()
    _row(idx, a, b, L, result, ms)
    print(SSEP)
    _detail(result)
 
    if mode == "plot":
        _launch_plot(idx, func, a, b, L, result)
    elif mode == "step":
        _launch_step(idx, func, a, b, L, result, tol)
 
 
def _launch_plot(idx, func, a, b, L, result):
    try:
        import matplotlib.pyplot as plt
        from visualizer import plot_result
        tf  = TestFunction(name=f"f{idx}", func=func, a=a, b=b, L=L)
        fig = plot_result(tf, result)
        plt.show()
    except ImportError as e:
        print(f"\n  [ATTENZIONE] matplotlib non disponibile: {e}")
 
 
def _launch_step(idx, func, a, b, L, result, tol):
    try:
        import matplotlib.pyplot as plt
        from visualizer import step_visualizer
        tf = TestFunction(name=f"f{idx}", func=func, a=a, b=b, L=L)
        step_visualizer(tf, result)
    except ImportError as e:
        print(f"\n  [ATTENZIONE] matplotlib non disponibile: {e}")
 
 
# ── Run su tutte le funzioni ──────────────────────────────────────────────────────────────
 
def run_allfunctions(tol: float, max_iter: int) -> None:
    _header()
    total_eval, total_ms = 0, 0.0
 
    for idx, (func, t) in TEST_REGISTRY.items():
        a, b, L = t
        t0      = time.perf_counter()
        res     = piShAlgorithm(func, a, b, L, tol=tol, max_iter=max_iter)
        ms      = (time.perf_counter() - t0) * 1000
        total_eval += res.n_evals
        total_ms   += ms
        _row(idx, a, b, L, res, ms)
 
    print(SSEP)
    n = len(TEST_REGISTRY)
    print(f"\n  Valutazioni totali : {total_eval}")
    print(f"  Tempo totale       : {total_ms:.2f} ms")
    print(f"  Tempo medio        : {total_ms / n:.2f} ms/funzione")
    print(SEP)
 
 
# ── Lista funzioni ─────────────────────────────────────────────────────────
 
def list_functions() -> None:
    print(f"\n  {'#':>3}   {'[a, b]':^26}   {'L':>10}")
    print("  " + "-" * 44)
    for idx, (_, t) in TEST_REGISTRY.items():
        a, b, L = t
        print(f"  {idx:>3}   [{a:>7}, {b:>7}]      {L:>10.2f}")
    print()
 
 
# ── Menu interattivo ───────────────────────────────────────────────────────
 
def _ask(prompt: str, cast=str, valid=None):
    while True:
        try:
            v = cast(input(prompt).strip())
        except (ValueError, TypeError):
            print("  Valore non valido, riprova.")
            continue
        if valid is not None and v not in valid:
            print(f"  Valori ammessi: {sorted(valid)}")
            continue
        return v
 
 
def _yes(prompt: str) -> bool:
    return input(prompt).strip().lower() in ("s", "si", "y", "yes")
 
 
def interactive_menu() -> None:
    print(SEP)
    print("   PIYAVSKI-SHUBERT — menu interattivo")
    print(SEP)
    print("\n  [1] Esegui su una funzione di test (f1-f20)")
    print("  [2] Esegui su tutte e 20 le funzioni")
    print("  [3] Lista delle funzioni disponibili")
    print("  [0] Esci")
    choice = _ask("\n  Scelta: ", cast=int, valid={0, 1, 2, 3})
 
    if choice == 0:
        return
    if choice == 3:
        list_functions()
        return
 
    tol = (
        _ask("  Tolleranza: ", cast=float)
        if _yes("  Personalizzare tolleranza? (default 1e-4) [s/N] ")
        else 1e-4
    )
    max_iter = (
        _ask("  Max iterazioni: ", cast=int)
        if _yes("  Personalizzare max_iter? (default 10000) [s/N] ")
        else 10_000
    )
 
    if choice == 2:
        run_allfunctions(tol, max_iter)
        return
 
    list_functions()
    idx       = _ask("  Numero funzione (1-20): ", cast=int, valid=set(range(1, 21)))
    func, t   = TEST_REGISTRY[idx]
    a, b, L   = t
 
    if _yes(f"  Sovrascrivere i parametri default [a={a}, b={b}, L={L}]? [s/N] "):
        a = _ask(f"    a [{a}]: ", cast=float)
        b = _ask(f"    b [{b}]: ", cast=float)
        L = _ask(f"    L [{L}]: ", cast=float)
 
    print("\n  Modalità output:")
    print("  [text] solo testo (default)")
    print("  [plot] grafico statico completo")
    print("  [step] navigatore passo-passo interattivo")
    mode = _ask("  Scelta [text/plot/step]: ", cast=str, valid={"text", "plot", "step"})
 
    run_single(idx, func, a, b, L, tol, max_iter, mode)
 
 
# ── CLI ────────────────────────────────────────────────────────────────────
 
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Algoritmo di Piyavski-Shubert — minimizzazione globale 1D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--fn",        type=int,   metavar="N",
                   help="Indice funzione di test (1-20)")
    p.add_argument("--a",         type=float, metavar="FLOAT",
                   help="Estremo sinistro (sovrascrive il default)")
    p.add_argument("--b",         type=float, metavar="FLOAT",
                   help="Estremo destro (sovrascrive il default)")
    p.add_argument("--L",         type=float, metavar="FLOAT",
                   help="Costante di Lipschitz (sovrascrive il default)")
    p.add_argument("--tol",       type=float, default=1e-4, metavar="FLOAT",
                   help="Tolleranza (default: 1e-4)")
    p.add_argument("--max-iter",  type=int,   default=10_000, metavar="INT",
                   help="Max iterazioni (default: 10000)")
    p.add_argument("--mode",      choices=["text", "plot", "step"], default="text",
                   help="Modalità output: text | plot | step  (default: text)")
    p.add_argument("--allFunctions", action="store_true",
                   help="Run su tutte e 20 le funzioni")
    p.add_argument("--list",      action="store_true",
                   help="Lista delle funzioni disponibili")
    return p
 
 
def main() -> None:
    if len(sys.argv) == 1:
        interactive_menu()
        return
 
    parser = _build_parser()
    args   = parser.parse_args()
 
    if args.list:
        list_functions()
        return
 
    if args.allFunctions:
        run_allfunctions(args.tol, args.max_iter)
        return
 
    if args.fn is None:
        parser.error("Specifica --fn N, oppure --allFunctions, oppure --list.")
    if args.fn not in TEST_REGISTRY:
        parser.error(f"--fn deve essere 1-20 (ricevuto: {args.fn}).")
 
    func, t = TEST_REGISTRY[args.fn]
    a = args.a if args.a is not None else t[0]
    b = args.b if args.b is not None else t[1]
    L = args.L if args.L is not None else t[2]
 
    run_single(args.fn, func, a, b, L, args.tol, args.max_iter, args.mode)
 
 
if __name__ == "__main__":
    main()
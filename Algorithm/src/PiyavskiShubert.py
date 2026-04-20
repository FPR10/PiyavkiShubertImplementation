"""
Algoritmo di Piyavki-Shuber per la minimizzazione globale di funzione univariata con costante di Lipschitz (L) nota.

Struttura dati: heapmin di oggetti Candidate (lower_bound, x_hat, x_left, x_right, f_left, f_right)
|
|--> ordina le tuple in modo lessicografico, quindi opera i confronti per lower_bound (primo elemento)
"""


'''
IMPORT
'''
import heapq 
from dataclasses import dataclass
from typing import Callable, List
import sys



'''
CLASSI DI SUPPORTO
'''


@dataclass(order=True)  #ordinamento su lb
class Candidate:
    """
    Rappresenta un sotto-intervallo [x_left, x_right].
 
    Attributi
    ---------
    lower_bound : chiave del min-heap — caratteristica R dell'intervallo.
    x_hat       : prossimo punto di valutazione (intersezione delle due rette di Lipschitz partenti dagli estremi).
    x_left      : estremo sinistro.
    x_right     : estremo destro.
    f_left      : f(x_left).
    f_right     : f(x_right).
    """ 
    lower_bound: float
    x_hat: float
    x_left: float
    x_right: float
    func_left: float
    func_right: float
    
    
def build_candidate(x_left: float, x_right: float, f_left: float, f_right: float, L: float) -> Candidate:
    """Costruisce un Candidate per l'intervallo [x_left, x_right] calcolando x_hat e lower_bound(caratteristica R)

    Args:
        x_left (float): estremo sx dell'intervallo
        x_right (float): estremo dx dell'intervallo
        f_left (float): valore funzione nell'estremo sx dell'intervallo
        f_right (float): valore funzione nell'estremo dx dell'intervallo
        L (float): costante di Lipschitz
        
    Calculate:
    
        x_hat: intersezione delle due rette di Lipschitz (prossimo punto di valutazione)
        lb: caratteristica R dell'intervallo (lower bound, chiave del min-heap)

    Returns:
        Candidate
    """
    x_hat = (x_left+x_right)/2 - (f_right-f_left)/(2*L)
    lb = (f_left+f_right)/2 - L*(x_right-x_left)/2
    return Candidate(lb, x_hat, x_left, x_right, f_left, f_right)


class PiyavskiShubertRes:
    """Incapsula i risultati restituiti dall'algoritmo
    
    Attributes:
        x_opt (float): punto x* in cui è stato trovato il minimo approssimato.
        f_opt (float): valore f(x*), ovvero il minimo trovato.
        iterations (int): numero di iterazioni eseguite dall'algoritmo.
        evalutations (int): numero totale di valutazioni di f, incluse le due iniziali in a e b.
        
        
    Attributes for plotting.py
        history (list[tuple[float, float]]): sequenza cronologica dei punti valutati come coppie (x, f(x)).
        candidates_log (list[list[Candidate]]) snapshot della heap ad ogni iterazione.
        
    
    """
    def __init__(self, x_opt, f_opt, iterations, evaluations, history, candidates_log):
        self.x_opt         = x_opt
        self.f_opt         = f_opt
        self.iterations    = iterations
        self.n_evals       = evaluations
        self.history       = history
        self.candidates_log = candidates_log

    def __repr__(self):
        return (f"PSResult(x_opt={self.x_opt:.6f}, f_opt={self.f_opt:.6f}, "
                f"iterations={self.iterations}, n_evals={self.n_evals})")
    


"""
ALGORITMO
"""
def piShAlgorithm(f:Callable[[float],float], a:float, b:float, L:float, tol:float = 1e-4, max_iter: int= 1000, store_candidates: bool = False) -> PiyavskiShubertRes:
    """
    Algoritmo iterativo di Piyavski-Shubert per la minimizzazione globale
    di una funzione f Lipschitz-continua su un intervallo [a, b].

    Args:
        f (Callable[[float], float]): Funzione obiettivo da minimizzare.
        a (float): Estremo sinistro dell'intervallo di ricerca.
        b (float): Estremo destro dell'intervallo di ricerca.
        L (float): Sovrastima della costante di Lipschitz. Deve essere L > 0.
                   Un valore troppo piccolo rischia di escludere regioni con
                   minimi globali; un valore troppo grande rallenta la convergenza.
        
        tol (float, optional): Tolleranza sul criterio d'arresto. L'algoritmo
                               si ferma quando la larghezza dell'intervallo con
                               caratteristica minima è <= tol. Defaults to 1e-4.
        
        max_iter (int, optional): Numero massimo di iterazioni consentite.
                                  Criterio d'arresto alternativo a tol.
                                  Defaults to 1000.
        
        store_candidates (bool, optional): Se True, salva uno snapshot della
                                           heap ad ogni iterazione in
                                           PSResult.candidates_log, utile per
                                           il plotting. Ha un costo in memoria.
                                           Defaults to False.

    Returns:
        PSResult: Oggetto contenente il minimo trovato, il punto ottimo,
                  il numero di iterazioni e valutazioni, la storia dei punti
                  valutati e (opzionalmente) il log dei candidati.
    """
    
    #Check parametri
    if L <= 0:
        raise ValueError("La costante di Lipschitz deve essere positivo")
    if a >= b:
        raise ValueError("a deve essere < b")
    
    #Precisione macchina per confronti robusti tra numeri reali
    eps = sys.float_info.epsilon
    
    
    '''
    INIZIALIZZAZIONE 
    k=1 -> ho un solo sotto-intervallo in [a,b]
    '''
    
    fa = f(a)
    fb = f(b)
    evaluations = 2
    
    heap: List[Candidate] = []
    heapq.heappush(heap, build_candidate(a,b,fa,fb,L))
    
    #Confronto tra due reali. Sono uguali se |fa-fb|<eps
    if fa-fb < eps:
        x_opt, f_opt = a, fa
    else:
        x_opt, f_opt = b, fb
    
    
    #Contiene [iterazione, x_opt, f_opt]
    history = [(0, x_opt, f_opt)]
    candidates_log = []
    
    
    for iteration in range(1, max_iter+1):
        
        if not heap:
            break
        
        """Passo 3: selezione del sotto-intervallo t con caratteristica minima Rt
        Con heap siamo sicuri di scegliere il candidato con lb minimo in O(log n)
        """
        best = heapq.heappop(heap)
        
        
        """Passo 4: criterio di arresto.
        Si ferma se la differenza tra f_opt e lb <= tol (non possiamo migliorare)
        """
        if f_opt - best.lower_bound <= tol:
            break
        
        """Passo 5: nuova valutazione della funzione in x_hat
        """
        x_new = max (best.x_left, min(best.x_right, best.x_hat))
        f_new = f(x_new)
        evaluations += 1
        
        
        """Aggiorno con l'ottimo migliore trovato
        """
        if f_new < f_opt - eps:
            x_opt, f_opt = x_new, f_new
            
            
        """Passo 1 + Passo 2: suddivisione dell'intervallo e calcolo delle nuove caratteristiche
        x_new suddivide [x_left, x_right] in due nuovi sotto-intervallo
        """
        left_cand = build_candidate(best.x_left, x_new, best.func_left, f_new, L)
        right_cand = build_candidate(x_new, best.x_right, f_new, best.func_right, L)
        
        
        """Ottimizzazione (pruning)
        Si inserisce il candidato nell'heap solo se la sua caratteristica lb è strettamente minore dell'ottimo corrente f_opt
        """
        if left_cand.lower_bound < f_opt - eps:
            heapq.heappush(heap, left_cand)

        if right_cand.lower_bound < f_opt - eps:
            heapq.heappush(heap, right_cand)

        history.append((iteration, x_opt, f_opt))

        if store_candidates:
            candidates_log.append(list(heap))
            
        
    return PiyavskiShubertRes(x_opt, f_opt, iteration, evaluations, history, candidates_log)
        
    
    

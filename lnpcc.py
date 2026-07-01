# -*- coding: utf-8 -*-
"""
LN-PCC em convenção Python:
-1 = não rotulado, 0..C-1 = classes.

Parâmetros principais:
- build_graph(data, slabel, k_nn=10):
    slabel: -1 = não rotulado, 0..C-1 = classes.
- fit_predict(labels, ...):
    labels: -1 = não rotulado, 0..C-1 = classes.
    Retorna rótulos em 0..C-1.

Opções extras:
- uniform_labeled (bool, default=False):
    False: nós rotulados começam one-hot (modo clássico, compatível com artigo).
    True : nós rotulados começam uniformes (versão mais robusta a ruído).
- dexp (float, default=2.0):
    Expoente do termo de distância: 1 / (1 + d)^dexp.

Backends:
- impl="numpy": usa lnpcc_step_numpy (referência pura NumPy).
- impl="numba": usa lnpcc_step_numba (Numba JIT).
- impl="cython": usa lnpcc_step (módulo Cython).
"""

import numpy as np
import warnings
from dataclasses import dataclass

from lnpcc_graph import build_knn_graph_with_labels
from lnpcc_numpy import lnpcc_step_numpy

try:
    from lnpcc_numba import lnpcc_step_numba, lnpcc_propagate_numba
    _HAS_NUMBA = True
except ImportError:
    lnpcc_step_numba = None
    lnpcc_propagate_numba = None
    _HAS_NUMBA = False

try:
    from lnpcc_step import lnpcc_step as lnpcc_step_cython
    from lnpcc_step import lnpcc_propagate_cython
    _HAS_CYTHON = True
except ImportError:
    lnpcc_step_cython = None
    lnpcc_propagate_cython = None
    _HAS_CYTHON = False


@dataclass
class ParticlesLN:
    homenode: np.ndarray
    curnode: np.ndarray
    label: np.ndarray
    strength: np.ndarray
    amount: int
    dist_table: np.ndarray


@dataclass
class NodesLN:
    amount: int
    dominance: np.ndarray
    label: np.ndarray


class LabelNoisePCC:  # ← classe começa aqui
    """
    LN-PCC: variante do PCC para rótulos ruidosos, com 3 mudanças:
      - grafo via build_knn_graph_with_labels;
      - nós rotulados também têm potenciais atualizados (não são congelados);
      - roda n_repeats vezes e acumula o potencial.

    Convenção externa: labels/slabel em -1 = não rotulado, 0..C-1 = classes.

    Opções:
      - uniform_labeled:
          False (default): nós rotulados começam one-hot (compatível com artigo).
          True            : nós rotulados começam uniformes (mais robusto a ruído).
      - dexp:
          Expoente do termo de distância 1/(1+d)^dexp. Default: 2.0.
      - impl:
          'numpy', 'numba' ou 'cython' para escolher backend.
    """

    def __init__(self, impl="auto", n_jobs=None):  # ← indentado dentro da classe
        if impl == "auto":
            if _HAS_CYTHON:
                self.impl = "cython"
            elif _HAS_NUMBA:
                self.impl = "numba"
            else:
                self.impl = "numpy"
        else:
            self.impl = impl
        self.n_jobs = n_jobs

        self.data = None
        self.k_nn = None
        self.neib_list = None
        self.neib_qt = None
        self.labels = None
        self.unique_labels = None
        self.c = None
        self.p_grd = None
        self.delta_v = None
        self.deltap = None
        self.dexp = None
        self.max_iter = None
        self.early_stop = None
        self.es_chk = None
        self.part = None
        self.node = None
        self.zerovec = None
        self.owndeg = None
        self.label_to_idx = None
        self.idx_to_label = None
        self.uniform_labeled = False

    def _propagate_backend(self,
                          neib_list, neib_qt,
                          labels, p_grd, delta_v, c, zerovec, dexp,
                          part_curnode, part_label, part_strength, dist_table,
                          dominance, owndeg, deltap,
                          dom_row, reduc, dom_list, dist_list, prob, slices,
                          dist_weights, max_iter, early_stop, es_chk, stop_max):

        backends = [self.impl] if self.impl != "auto" else ["cython", "numba", "numpy"]

        for be in backends:
            if be == "cython" and _HAS_CYTHON:
                try:
                    lnpcc_propagate_cython(neib_list, neib_qt,
                                          labels, p_grd, delta_v, c,
                                          zerovec,
                                          part_curnode, part_label, part_strength, dist_table,
                                          dominance, owndeg, deltap,
                                          np.ascontiguousarray(dist_weights),
                                          np.ascontiguousarray(prob),
                                          np.ascontiguousarray(slices),
                                          max_iter, int(early_stop), es_chk, stop_max)
                    return True
                except Exception as e:
                    import warnings
                    warnings.warn(f"Cython propagation failed with error: {e}. Falling back.")

            if be == "numba" and _HAS_NUMBA:
                try:
                    lnpcc_propagate_numba(neib_list, neib_qt,
                                         labels, p_grd, delta_v, c,
                                         zerovec,
                                         part_curnode, part_label, part_strength, dist_table,
                                         dominance, owndeg, deltap, dexp,
                                         dom_row, reduc, dom_list, dist_list, prob, slices,
                                         dist_weights, max_iter, early_stop, es_chk, stop_max)
                    return True
                except Exception as e:
                    import warnings
                    warnings.warn(f"Numba propagation failed with error: {e}. Falling back.")

        return False

    def _step_backend(self,
                      neib_list, neib_qt,
                      labels, p_grd, delta_v, c, zerovec, dexp,
                      part_curnode, part_label, part_strength, dist_table,
                      dominance, owndeg, deltap,
                      dom_row, reduc, dom_list, dist_list, prob, slices, dist_weights):
        # Keep old step backend for fallback or individual steps
        backends = [self.impl] if self.impl != "auto" else ["cython", "numba", "numpy"]

        for be in backends:
            if be == "cython" and _HAS_CYTHON:
                lnpcc_step_cython(neib_list, neib_qt,
                                  labels, p_grd, delta_v, c,
                                  zerovec, dexp,
                                  part_curnode, part_label, part_strength, dist_table,
                                  dominance, owndeg, deltap,
                                  prob, slices, dist_weights)
                return

            if be == "numba" and _HAS_NUMBA:
                lnpcc_step_numba(neib_list, neib_qt,
                                 labels, p_grd, delta_v, c,
                                 zerovec, dexp,
                                 part_curnode, part_label, part_strength, dist_table,
                                 dominance, owndeg, deltap,
                                 dom_row=dom_row, reduc=reduc,
                                 dom_list=dom_list, dist_list=dist_list,
                                 prob=prob, slices=slices)
                return

            if be == "numpy":
                lnpcc_step_numpy(neib_list, neib_qt,
                                 labels, p_grd, delta_v, c,
                                 zerovec, dexp,
                                 part_curnode, part_label, part_strength, dist_table,
                                 dominance, owndeg, deltap,
                                 dom_row=dom_row, reduc=reduc,
                                 dom_list=dom_list, dist_list=dist_list,
                                 prob=prob, slices=slices)
                return

    def build_graph(self, data, slabel, k_nn=10, n_jobs=None):
        """
        data: X (features)
        slabel: -1 = não rotulado, 0..C-1 = classes.
        """
        slabel = np.asarray(slabel, dtype=np.int64)

        self.data = data
        self.k_nn = k_nn
        self.neib_list, self.neib_qt = build_knn_graph_with_labels(
            data, slabel, k_nn, n_jobs=n_jobs if n_jobs is not None else self.n_jobs
        )

    def set_graph(self, neib_list, neib_qt, k_nn=None):
        """
        Define explicitamente o grafo (por exemplo, vindo do PyG).
        """
        self.neib_list = np.asarray(neib_list, dtype=np.int64)
        self.neib_qt = np.asarray(neib_qt, dtype=np.int64)
        self.k_nn = int(k_nn) if k_nn is not None else int(self.neib_qt.max())
        self.data = None

    def fit_predict(self, labels, p_grd=0.5, delta_v=0.1, dexp=2.0, deltap=1.0,
                    max_iter=500000, early_stop=True, es_chk=2000,
                    n_repeats=10, uniform_labeled=False):
        """
        labels: -1 = não rotulado, 0..C-1 = classes.
        Retorna rótulos em 0..C-1.

        uniform_labeled:
          False: nós rotulados começam one-hot (modo clássico, artigo).
          True : nós rotulados começam uniformes (modo robusto).

        dexp:
          Expoente da distância na escolha greedy: 1 / (1 + d)^dexp.
        """
        if self.neib_list is None or self.neib_qt is None:
            print("Error: You must build or set the graph first "
                  "using build_graph(data, slabel) or set_graph(neib_list, neib_qt).")
            return -1

        self.labels = np.ascontiguousarray(labels, dtype=np.int64)
        
        # Internal label mapping: maps original labels to 0..c-1
        self.unique_labels = np.unique(self.labels)
        self.unique_labels = self.unique_labels[self.unique_labels != -1]
        self.c = len(self.unique_labels)
        
        if self.c < 2:
            import warnings
            warnings.warn("LN-PCC requires at least 2 labeled classes to run. Returning original labels.")
            return labels
        
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.unique_labels)}
        self.idx_to_label = {i: lbl for i, lbl in enumerate(self.unique_labels)}
        
        # Create a mapped labels array for internal use
        self.mapped_labels = np.full(self.labels.shape, -1, dtype=np.int64)
        for lbl, idx in self.label_to_idx.items():
            self.mapped_labels[self.labels == lbl] = idx
        self.mapped_labels = np.ascontiguousarray(self.mapped_labels)

        self.p_grd = float(p_grd)
        self.delta_v = float(delta_v)
        self.deltap = float(deltap)
        self.dexp = float(dexp)
        self.max_iter = int(max_iter)
        self.early_stop = bool(early_stop)
        self.es_chk = int(es_chk)
        self.uniform_labeled = bool(uniform_labeled)

        qtnode = self.neib_list.shape[0]
        n_classes = self.c

        potacc = np.zeros((qtnode, n_classes), dtype=np.float64)
        owndeg_acc = np.zeros((qtnode, n_classes), dtype=np.float64)

        for r_i in range(n_repeats):
            self.part = self.__genParticles()
            self.node = self.__genNodes()
            self.__labelPropagation()
            potacc += self.node.dominance
            owndeg_acc += self.owndeg

        owner_internal = np.argmax(potacc, axis=1)  # 0..C-1
        
        # Unmap labels before returning
        final_labels = np.full(owner_internal.shape, -1, dtype=np.int64)
        for idx, lbl in self.idx_to_label.items():
            final_labels[owner_internal == idx] = lbl

        # Normaliza owndeg acumulado
        row_sums = np.sum(owndeg_acc, axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        self.owndeg = np.ascontiguousarray(owndeg_acc / row_sums)

        return final_labels

    def __labelPropagation(self):
        self.zerovec = np.zeros(self.c, dtype=np.float64)

        # Pre-allocate buffers for performance and safety
        # Use the actual maximum degree of the graph, which may be larger than nominal k_nn due to symmetrization
        max_deg = max(int(self.neib_qt.max()), 1)
        self.buf_dom_row = np.empty(self.c, dtype=np.float64)
        self.buf_reduc = np.empty(self.c, dtype=np.float64)
        self.buf_dom_list = np.empty(max_deg, dtype=np.float64)
        self.buf_dist_list = np.empty(max_deg, dtype=np.float64)
        self.buf_prob = np.empty(max_deg, dtype=np.float64)
        self.buf_slices = np.empty(max_deg, dtype=np.float64)

        early_stop = self.early_stop
        node = self.node
        part = self.part

        k_nn = self.k_nn if self.k_nn is not None else int(self.neib_qt.max())

        es_chk = self.es_chk
        max_iter = self.max_iter
        neib_list = self.neib_list
        neib_qt = self.neib_qt

        # stop_max must always be defined (used by backend even when early_stop=False).
        # Clamp to at least 1 so the early-stop logic doesn't fire on the very first check
        # (common on large graphs like amazon-ratings where the ratio becomes < 0.5).
        stop_max = max(1, round((node.amount / (part.amount * k_nn)) * round(es_chk * 0.1)))
        if early_stop:
            max_mmpot = 0.0
            stop_cnt = 0

        # Use pre-allocated buffers from __labelPropagation
        dom_row = self.buf_dom_row
        reduc = self.buf_reduc
        dom_list = self.buf_dom_list
        dist_list = self.buf_dist_list
        prob = self.buf_prob
        slices = self.buf_slices

        # Pre-calculate distance weights
        dist_weights = np.ascontiguousarray(
            1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** self.dexp,
            dtype=np.float64
        )

        # Use explicit contiguous arrays for the backend to avoid alignment issues on Windows
        success = self._propagate_backend(
            neib_list, neib_qt,
            self.mapped_labels, self.p_grd, self.delta_v, self.c,
            self.zerovec, self.dexp,
            part.curnode, part.label,
            part.strength, part.dist_table,
            node.dominance, self.owndeg, self.deltap,
            dom_row, reduc,
            dom_list, dist_list,
            prob, slices,
            dist_weights, max_iter, early_stop, es_chk, stop_max
        )

        if not success:
            # Fallback to Python loop with individual steps
            for it in range(max_iter):
                self._step_backend(neib_list, neib_qt,
                                   self.mapped_labels, self.p_grd, self.delta_v, self.c,
                                   self.zerovec, self.dexp,
                                   part.curnode, part.label, part.strength, part.dist_table,
                                   node.dominance, self.owndeg, self.deltap,
                                   dom_row, reduc, dom_list, dist_list, prob, slices, dist_weights)

                if early_stop and it % 10 == 0:
                    mmpot = np.mean(np.amax(node.dominance, 1))
                    if mmpot > max_mmpot:
                        max_mmpot = mmpot
                        stop_cnt = 0
                    else:
                        stop_cnt += 1
                        if stop_cnt > stop_max:
                            break

        # apenas para depuração: preenche rótulos de nós originalmente não rotulados
        unlabeled = node.label == -1
        node.label[unlabeled] = np.argmax(node.dominance[unlabeled, :], axis=1)

    def __genParticles(self) -> ParticlesLN:
        homenode = np.where(self.mapped_labels != -1)[0].astype(np.int64)
        curnode = homenode.copy()
        label = self.mapped_labels[self.mapped_labels != -1].astype(np.int64)
        amount = int(homenode.shape[0])
        strength = np.full(amount, 1.0, dtype=np.float64)

        n_nodes = self.neib_list.shape[0]
        max_dist = min(n_nodes - 1, 255)
        dist_table = np.full(
            shape=(n_nodes, amount),
            fill_value=max_dist,
            dtype=np.uint8,
        )
        dist_table[homenode, np.arange(amount)] = 0

        return ParticlesLN(
            homenode=homenode,
            curnode=curnode,
            label=label,
            strength=strength,
            amount=amount,
            dist_table=dist_table,
        )

    def __genNodes(self) -> NodesLN:
        amount = self.neib_list.shape[0]
        n_classes = self.c

        # Inicialização padrão: tudo uniforme
        dominance = np.full(
            shape=(amount, n_classes),
            fill_value=float(1.0 / n_classes),
            dtype=np.float64,
        )

        label = self.mapped_labels.copy().astype(np.int64)
        dominance[label != -1, :] = 0.0

        for l_idx in range(self.c):
            dominance[label == l_idx, l_idx] = 1.0

        if self.uniform_labeled:
            # Sobrescreve para uniforme se solicitado
            labeled_idx = np.where(label != -1)[0]
            dominance[labeled_idx, :] = 1.0 / n_classes

        # owndeg começa quase zero (realmin) e é acumulado ao longo da propagação.
        self.owndeg = np.full(
            shape=(amount, n_classes),
            fill_value=np.finfo(float).tiny,
            dtype=np.float64,
        )

        return NodesLN(
            amount=amount,
            dominance=dominance,
            label=label,
        )

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 16:14:35 2026

@author: fbrev
"""

# lnpcc_numpy.py
import numpy as np

def lnpcc_step_numpy(neib_list, neib_qt,
                     labels, p_grd, delta_v, c, zerovec, dexp,
                     part_curnode, part_label, part_strength, dist_table,
                     dominance, owndeg, deltap=1.0,
                     dom_row=None, reduc=None, dom_list=None, dist_list=None, prob=None, slices=None):
    """
    Versão NumPy puro do _lnpcc_step com suporte a buffers externos e blindagem de memória.
    """
    n_particles = part_curnode.shape[0]
    n_nodes = neib_list.shape[0]

    # fallback allocation if buffers not passed
    if dom_row is None:
        dom_row = np.empty(c, dtype=np.float64)
    if dom_list is None:
        m_k = neib_qt.max() if neib_qt.size > 0 else 1
        dom_list = np.empty(m_k, dtype=np.float64)
        dist_list = np.empty(m_k, dtype=np.float64)
        prob = np.empty(m_k, dtype=np.float64)
        slices = np.empty(m_k, dtype=np.float64)

    for p_i in range(n_particles):
        curnode = part_curnode[p_i]
        if curnode < 0 or curnode >= n_nodes:
            continue
            
        k = neib_qt[curnode]
        if k <= 0:
            continue
            
        neighbors = neib_list[curnode, :k]

        if np.random.random() < p_grd:
            greedy = 1
            label = part_label[p_i]
            
            # class dominance of the particle in each neighbor (guarded)
            if label >= 0 and label < c:
                for i in range(k):
                    nx_node = neighbors[i]
                    dom_list[i] = dominance[nx_node, label]
            else:
                dom_list[:k] = 0.0

            # dist_list: 1 / (1 + d)^dexp
            for i in range(k):
                nx_node = neighbors[i]
                d_val = float(dist_table[nx_node, p_i])
                dist_list[i] = 1.0 / ((1.0 + d_val) ** dexp)

            p_move = dom_list[:k] * dist_list[:k]
            p_sum = np.sum(p_move)

            if p_sum <= 0.0:
                choice = np.random.randint(0, k)
            else:
                p_move /= p_sum
                choice = np.random.choice(k, p=p_move)
            next_node = neighbors[choice]
        else:
            greedy = 0
            choice = np.random.randint(0, k)
            next_node = neighbors[choice]

        # Validar label novamente
        label = part_label[p_i]

        # update dominance/strength/pos
        if next_node >= 0 and next_node < n_nodes and label >= 0 and label < c:
            # step 1: calculate reduction
            dom_row_local = dominance[next_node, :].copy()
            
            # Division by zero safety (though lnpcc.py checks c<2)
            c_m1 = float(c - 1) if c > 1 else 1.0
            step = part_strength[p_i] * (delta_v / c_m1)
            
            reduc_local = dom_row_local - np.maximum(dom_row_local - step, 0.0)
            reduc_sum = np.sum(reduc_local)
            
            # step 2: apply update
            dom_row_local -= reduc_local
            dom_row_local[label] += reduc_sum
            dominance[next_node, :] = dom_row_local

            # strength
            part_strength[p_i] += (dominance[next_node, label] - part_strength[p_i]) * deltap

            # HARDENING: distance update with overflow protection (prevent uint8 wrap 255 -> 0)
            cur_d = dist_table[curnode, p_i]
            next_d = dist_table[next_node, p_i]
            if cur_d < 255:
                if next_d > cur_d + 1:
                    dist_table[next_node, p_i] = cur_d + 1

            if greedy == 0:
                owndeg[next_node, label] += part_strength[p_i]

            # shock
            if dominance[next_node, label] == np.max(dominance[next_node, :]):
                part_curnode[p_i] = next_node
        else:
            pass

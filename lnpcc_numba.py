# -*- coding: utf-8 -*-
"""
Optimized LN-PCC kernel in Numba (Backend-Loop architecture).
Created on Fri Jan 16 16:14:55 2026
@author: fbrev
"""

import numpy as np
from numba import njit

@njit
def _lnpcc_calc_mmpot(dominance, n_nodes, c):
    """
    Sequential version of mean amax.
    """
    total_max = 0.0
    for i in range(n_nodes):
        row_max = dominance[i, 0]
        for j in range(1, c):
            if dominance[i, j] > row_max:
                row_max = dominance[i, j]
        total_max += row_max
    return total_max / n_nodes

@njit
def _lnpcc_step_numba(neib_list, neib_qt,
                      labels, p_grd, delta_v, c, zerovec,
                      part_curnode, part_label, part_strength, dist_table,
                      dominance, owndeg, deltap,
                      dom_row, reduc, dom_list, dist_list, prob, slices,
                      dist_weights):
    n_particles = part_curnode.shape[0]
    n_nodes = neib_list.shape[0]

    for p_i in range(n_particles):
        curnode = part_curnode[p_i]
        if curnode < 0 or curnode >= n_nodes:
            continue

        k = neib_qt[curnode]
        if k <= 0:
            continue

        label = part_label[p_i]
        use_greedy = np.random.random() < p_grd
        
        if use_greedy:
            prob_sum = 0.0
            for i in range(k):
                nx_node = neib_list[curnode, i]
                
                if nx_node >= 0 and nx_node < n_nodes and label >= 0 and label < c:
                    d_idx = dist_table[nx_node, p_i]
                    # lookup instead of calculation
                    prob_val = dist_weights[d_idx] * dominance[nx_node, label]
                else:
                    prob_val = 0.0
                
                prob[i] = prob_val
                prob_sum += prob_val
                slices[i] = prob_sum

            if prob_sum <= 0.0:
                choice = np.random.randint(0, k)
            else:
                randval_move = np.random.random() * prob_sum
                choice = 0
                for i in range(k):
                    if randval_move <= slices[i]:
                        choice = i
                        break
            next_node = neib_list[curnode, choice]
            greedy = True
        else:
            greedy = False
            choice = np.random.randint(0, k)
            next_node = neib_list[curnode, choice]

        # Validar label novamente antes do update
        if next_node >= 0 and next_node < n_nodes and label >= 0 and label < c:
            # Step 1: calculate reduction
            # Division by zero safety for c=1
            c_inv = 1.0 / float(c - 1) if c > 1 else 1.0
            reduction_step = part_strength[p_i] * (delta_v * c_inv)
            
            reduc_sum = 0.0
            for i in range(c):
                val = dominance[next_node, i]
                # reduc_val = min(val, reduction_step)
                if val < reduction_step:
                    reduc_val = val
                else:
                    reduc_val = reduction_step
                
                dominance[next_node, i] -= reduc_val
                reduc_sum += reduc_val
            
            # Step 2: apply total reduction to particle's class
            dominance[next_node, label] += reduc_sum

            # update strength
            if deltap == 1.0:
                part_strength[p_i] = dominance[next_node, label]
            else:
                part_strength[p_i] += (dominance[next_node, label] - part_strength[p_i]) * deltap

            # distance update
            cur_d = dist_table[curnode, p_i]
            next_d = dist_table[next_node, p_i]
            if cur_d < 255:
                if next_d > cur_d + 1:
                    dist_table[next_node, p_i] = cur_d + 1

            if not greedy:
                owndeg[next_node, label] += part_strength[p_i]

            # shock: move if class is dominant
            max_dom = dominance[next_node, 0]
            for i in range(1, c):
                if dominance[next_node, i] > max_dom:
                    max_dom = dominance[next_node, i]

            if dominance[next_node, label] == max_dom:
                part_curnode[p_i] = next_node

@njit
def lnpcc_propagate_numba(neib_list, neib_qt,
                          labels, p_grd, delta_v, c, zerovec,
                          part_curnode, part_label, part_strength, dist_table,
                          dominance, owndeg, deltap, dexp,
                          dom_row, reduc, dom_list, dist_list, prob, slices,
                          dist_weights,
                          max_iter, early_stop, es_chk, stop_max):
    
    n_nodes = neib_list.shape[0]
    max_mmpot = 0.0
    stop_cnt = 0
    
    for it in range(max_iter):
        _lnpcc_step_numba(neib_list, neib_qt,
                          labels, p_grd, delta_v, c, zerovec,
                          part_curnode, part_label, part_strength, dist_table,
                          dominance, owndeg, deltap,
                          dom_row, reduc, dom_list, dist_list, prob, slices,
                          dist_weights)
        
        if early_stop and it % 10 == 0:
            mmpot = _lnpcc_calc_mmpot(dominance, n_nodes, c)
            if mmpot > max_mmpot:
                max_mmpot = mmpot
                stop_cnt = 0
            else:
                stop_cnt += 1
                if stop_cnt > stop_max:
                    break

def lnpcc_step_numba(neib_list, neib_qt,
                     labels, p_grd, delta_v, c, zerovec, dexp,
                     part_curnode, part_label, part_strength, dist_table,
                     dominance, owndeg, deltap=1.0,
                     dom_row=None, reduc=None, dom_list=None, dist_list=None, prob=None, slices=None):
    """
    Backward compatibility wrapper for a single step.
    """
    if dom_row is None:
        dom_row = np.empty(c, dtype=np.float64)
        reduc = np.empty(c, dtype=np.float64)
        m_k = neib_qt.max()
        dom_list = np.empty(m_k, dtype=np.float64)
        dist_list = np.empty(m_k, dtype=np.float64)
        prob = np.empty(m_k, dtype=np.float64)
        slices = np.empty(m_k, dtype=np.float64)
    
    dist_weights = 1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** dexp

    return _lnpcc_step_numba(neib_list, neib_qt,
                             labels, p_grd, delta_v, c, zerovec,
                             part_curnode, part_label, part_strength, dist_table,
                             dominance, owndeg, deltap,
                             dom_row, reduc, dom_list, dist_list, prob, slices,
                             dist_weights)

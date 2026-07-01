# -*- coding: utf-8 -*-
"""
Optimized LN-PCC kernel in Cython (Backend-Loop architecture).
Created on Fri Jan 16 16:15:18 2026
@author: fbrev
"""

# distutils: language = c
# distutils: extra_compile_args = -O3 -march=native
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX, malloc, free

# Fast XORShift RNG
@cython.cdivision(True)
cdef inline unsigned int xorshift32(unsigned int *state) noexcept nogil:
    cdef unsigned int x = state[0]
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    state[0] = x
    return x

@cython.cdivision(True)
cdef inline double rand_double(unsigned int *state) noexcept nogil:
    cdef unsigned int x = xorshift32(state)
    return <double>x / 4294967295.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _lnpcc_calc_mmpot(double[:, ::1] dominance, Py_ssize_t n_nodes, Py_ssize_t c) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double row_max, total_max = 0.0
    for i in range(n_nodes):
        row_max = dominance[i, 0]
        for j in range(1, c):
            if dominance[i, j] > row_max:
                row_max = dominance[i, j]
        total_max += row_max
    return total_max / n_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _lnpcc_step_cython(
    const np.int64_t[:, ::1] neib_list,
    const np.int64_t[::1]    neib_qt,
    const np.int64_t[::1]    labels,
    double                   p_grd,
    double                   delta_v,
    np.int64_t               c,
    const double[::1]        zerovec,
    np.int64_t[::1]          part_curnode,
    np.int64_t[::1]          part_label,
    double[::1]              part_strength,
    np.uint8_t[:, ::1]       dist_table,
    double[:, ::1]           dominance,
    double[:, ::1]           owndeg,
    double                   deltap,
    const double[::1]        dist_weights,
    double[::1]              prob_buf,
    double[::1]              slices_buf,
    unsigned int            *rng_state
) noexcept nogil:
    cdef Py_ssize_t n_particles = part_curnode.shape[0]
    cdef Py_ssize_t n_nodes = neib_list.shape[0]
    cdef Py_ssize_t p_i, curnode, k, i, choice, next_node
    cdef Py_ssize_t label
    cdef double randval, step, reduc_val, sum_reduc
    cdef double prob_sum, max_dom
    cdef np.uint8_t cur_d, next_d
    cdef int greedy
    
    for p_i in range(n_particles):
        curnode = part_curnode[p_i]
        # Robust bounds check for current node index
        if curnode < 0 or curnode >= n_nodes: continue

        k = neib_qt[curnode]
        if k <= 0: continue

        label = part_label[p_i]
        # Robust bounds check for class label
        if label < 0 or label >= c: continue
            
        randval = rand_double(rng_state)

        if randval < p_grd:
            # greedy
            prob_sum = 0.0
            for i in range(k):
                next_node = neib_list[curnode, i]
                # lookup in dist_weights table with bounds check
                if next_node < 0 or next_node >= n_nodes:
                    prob_buf[i] = 0.0
                else:
                    # dist_table access with guaranteed bounds
                    cur_d = dist_table[next_node, p_i]
                    prob_buf[i] = dist_weights[cur_d] * dominance[next_node, label]
                
                prob_sum += prob_buf[i]
                slices_buf[i] = prob_sum
            
            if prob_sum > 0.0:
                randval = rand_double(rng_state) * prob_sum
                choice = 0
                for i in range(k):
                    if randval <= slices_buf[i]:
                        choice = i
                        break
                next_node = neib_list[curnode, choice]
                greedy = 1
            else:
                next_node = neib_list[curnode, xorshift32(rng_state) % k]
                greedy = 0
        else:
            # random
            next_node = neib_list[curnode, xorshift32(rng_state) % k]
            greedy = 0

        # update WITHOUT checking if node is labeled (LN-PCC rule)
        if next_node >= 0 and next_node < n_nodes:
            step = part_strength[p_i] * (delta_v / (<double>(c - 1) if c > 1 else 1.0))
            sum_reduc = 0.0
            for i in range(c):
                if dominance[next_node, i] < step:
                    reduc_val = dominance[next_node, i]
                else:
                    reduc_val = step
                dominance[next_node, i] -= reduc_val
                sum_reduc += reduc_val
            
            dominance[next_node, label] += sum_reduc
            
            # Numerical safety: ensure dominance stays in valid range [0, 1]
            for i in range(c):
                if dominance[next_node, i] < 1e-15:
                    dominance[next_node, i] = 1e-15
                elif dominance[next_node, i] > 1.0:
                    dominance[next_node, i] = 1.0
            
            if deltap == 1.0:
                part_strength[p_i] = dominance[next_node, label]
            else:
                part_strength[p_i] += (dominance[next_node, label] - part_strength[p_i]) * deltap
            
            # Extra safety for particle strength
            if part_strength[p_i] < 1e-15: part_strength[p_i] = 1e-15
            if part_strength[p_i] > 1.0: part_strength[p_i] = 1.0

            cur_d = dist_table[curnode, p_i]
            next_d = dist_table[next_node, p_i]
            if cur_d < 255:
                if next_d > <np.uint8_t>(cur_d + 1):
                    dist_table[next_node, p_i] = cur_d + 1

            if greedy == 0:
                owndeg[next_node, label] += part_strength[p_i]

            max_dom = dominance[next_node, 0]
            for i in range(1, c):
                if dominance[next_node, i] > max_dom:
                    max_dom = dominance[next_node, i]

            if dominance[next_node, label] == max_dom:
                part_curnode[p_i] = next_node

cpdef lnpcc_propagate_cython(
    np.int64_t[:, ::1] neib_list,
    np.int64_t[::1]    neib_qt,
    np.int64_t[::1]    labels,
    double             p_grd,
    double             delta_v,
    np.int64_t         c,
    double[::1]        zerovec,
    np.int64_t[::1]    part_curnode,
    np.int64_t[::1]    part_label,
    double[::1]        part_strength,
    np.uint8_t[:, ::1] dist_table,
    double[:, ::1]     dominance,
    double[:, ::1]     owndeg,
    double             deltap,
    double[::1]        dist_weights,
    double[::1]        prob,
    double[::1]        slices,
    int                max_iter,
    int                early_stop,
    int                es_chk,
    int                stop_max
):
    cdef Py_ssize_t it, n_nodes = neib_list.shape[0]
    cdef double mmpot, max_mmpot = 0.0
    cdef int stop_cnt = 0
    cdef unsigned int rng_state = 123456789
    
    with nogil:
        for it in range(max_iter):
            _lnpcc_step_cython(neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
                               part_curnode, part_label, part_strength, dist_table,
                               dominance, owndeg, deltap, dist_weights,
                               prob, slices,
                               &rng_state)

            if early_stop != 0 and it % 10 == 0:
                mmpot = _lnpcc_calc_mmpot(dominance, n_nodes, c)
                if mmpot > max_mmpot:
                    max_mmpot = mmpot
                    stop_cnt = 0
                else:
                    stop_cnt += 1
                    if stop_cnt > stop_max:
                        break

cpdef lnpcc_step(
    np.int64_t[:, ::1] neib_list,
    np.int64_t[::1]    neib_qt,
    np.int64_t[::1]    labels,
    double             p_grd,
    double             delta_v,
    np.int64_t         c,
    double[::1]        zerovec,
    double             dexp,
    np.int64_t[::1]    part_curnode,
    np.int64_t[::1]    part_label,
    double[::1]        part_strength,
    np.uint8_t[:, ::1] dist_table,
    double[:, ::1]     dominance,
    double[:, ::1]     owndeg,
    double             deltap = 1.0,
    double[::1]        prob = None,
    double[::1]        slices = None,
    double[::1]        dist_weights = None,
):
    cdef unsigned int rng_state = 123456789
    if dist_weights is None:
        dist_weights = np.ascontiguousarray(1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** dexp)
    
    with nogil:
        _lnpcc_step_cython(neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
                           part_curnode, part_label, part_strength, dist_table,
                           dominance, owndeg, deltap, dist_weights,
                           prob, slices,
                           &rng_state)


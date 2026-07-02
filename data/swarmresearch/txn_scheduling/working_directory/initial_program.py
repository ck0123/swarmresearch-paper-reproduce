import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

import time
import math
import os
import ctypes
import tempfile

def get_best_schedule(workload, num_seqs):
    """
    C-accelerated evaluator + guided IG + SA.
    Uses compiled C code for ~50-100x faster schedule evaluation.
    """
    n = workload.num_txns
    time_budget = 190  # seconds per workload
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    # Remap keys to compact indices
    all_keys = set()
    for txn_ops in workload.txns:
        for (op_type, key, pos, txn_len) in txn_ops:
            all_keys.add(int(key))
    key_to_idx = {k: i for i, k in enumerate(sorted(all_keys))}
    num_keys = len(key_to_idx)

    # Preprocess transactions into flat arrays for C
    txns_data = []  # Python version
    all_ops_flat = []  # flat list of (is_write, key_idx, pos, txn_len) for C
    txn_offsets = [0]  # start index of each txn in all_ops_flat
    txn_num_ops = []

    for i in range(n):
        ops = []
        for (op_type, key, pos, txn_len) in workload.txns[i]:
            is_w = 1 if op_type == 'w' else 0
            kidx = key_to_idx[int(key)]
            ops.append((is_w, kidx, pos, txn_len))
            all_ops_flat.extend([is_w, kidx, pos, txn_len])
        txns_data.append(tuple(ops))
        txn_num_ops.append(len(ops))
        txn_offsets.append(txn_offsets[-1] + len(ops))

    total_ops = txn_offsets[-1]

    # Compile C evaluator
    c_code = r"""
#include <string.h>
#include <stdlib.h>

// Evaluate a full schedule
int fast_eval_c(int* seq, int n, int* ops_flat, int* offsets, int* num_ops_arr, int num_keys) {
    int* latest_end = (int*)malloc(num_keys * sizeof(int));
    int* latest_is_write = (int*)malloc(num_keys * sizeof(int));
    int* last_write_time = (int*)malloc(num_keys * sizeof(int));

    memset(latest_end, -1, num_keys * sizeof(int));
    memset(latest_is_write, 0, num_keys * sizeof(int));
    memset(last_write_time, -1, num_keys * sizeof(int));

    int total_cost = 0;

    for (int s = 0; s < n; s++) {
        int txn_idx = seq[s];
        int start_op = offsets[txn_idx] * 4;
        int nops = num_ops_arr[txn_idx];
        int txn_start = 1;
        int txn_len = ops_flat[start_op + 3]; // txn_len from first op

        for (int j = 0; j < nops; j++) {
            int base = start_op + j * 4;
            int is_write = ops_flat[base];
            int key = ops_flat[base + 1];
            int pos = ops_flat[base + 2];

            int le = latest_end[key];
            if (le >= 0) {
                int constraint;
                if (latest_is_write[key] || is_write) {
                    constraint = le + 2 - pos;
                } else {
                    int lwt = last_write_time[key];
                    constraint = (lwt >= 0) ? (lwt + 2 - pos) : (1 - pos);
                }
                if (constraint > txn_start) {
                    txn_start = constraint;
                }
            }
        }

        int txn_end = txn_start + txn_len - 1;
        int cost_inc = txn_end - total_cost;
        if (cost_inc < 0) cost_inc = 0;
        total_cost += cost_inc;

        for (int j = 0; j < nops; j++) {
            int base = start_op + j * 4;
            int is_write = ops_flat[base];
            int key = ops_flat[base + 1];
            int pos = ops_flat[base + 2];
            int actual_time = txn_start + pos - 1;

            if (is_write) {
                latest_end[key] = actual_time;
                latest_is_write[key] = 1;
                last_write_time[key] = actual_time;
            } else {
                if (actual_time > latest_end[key]) {
                    latest_end[key] = actual_time;
                }
                latest_is_write[key] = 0;
            }
        }
    }

    free(latest_end);
    free(latest_is_write);
    free(last_write_time);
    return total_cost;
}

// Evaluate with per-position cost tracking
int fast_eval_with_costs_c(int* seq, int n, int* ops_flat, int* offsets, int* num_ops_arr, int num_keys, int* cost_out) {
    int* latest_end = (int*)malloc(num_keys * sizeof(int));
    int* latest_is_write = (int*)malloc(num_keys * sizeof(int));
    int* last_write_time = (int*)malloc(num_keys * sizeof(int));

    memset(latest_end, -1, num_keys * sizeof(int));
    memset(latest_is_write, 0, num_keys * sizeof(int));
    memset(last_write_time, -1, num_keys * sizeof(int));

    int total_cost = 0;

    for (int s = 0; s < n; s++) {
        int txn_idx = seq[s];
        int start_op = offsets[txn_idx] * 4;
        int nops = num_ops_arr[txn_idx];
        int txn_start = 1;
        int txn_len = ops_flat[start_op + 3];

        for (int j = 0; j < nops; j++) {
            int base = start_op + j * 4;
            int is_write = ops_flat[base];
            int key = ops_flat[base + 1];
            int pos = ops_flat[base + 2];

            int le = latest_end[key];
            if (le >= 0) {
                int constraint;
                if (latest_is_write[key] || is_write) {
                    constraint = le + 2 - pos;
                } else {
                    int lwt = last_write_time[key];
                    constraint = (lwt >= 0) ? (lwt + 2 - pos) : (1 - pos);
                }
                if (constraint > txn_start) {
                    txn_start = constraint;
                }
            }
        }

        int txn_end = txn_start + txn_len - 1;
        int cost_inc = txn_end - total_cost;
        if (cost_inc < 0) cost_inc = 0;
        total_cost += cost_inc;
        cost_out[s] = cost_inc;

        for (int j = 0; j < nops; j++) {
            int base = start_op + j * 4;
            int is_write = ops_flat[base];
            int key = ops_flat[base + 1];
            int pos = ops_flat[base + 2];
            int actual_time = txn_start + pos - 1;

            if (is_write) {
                latest_end[key] = actual_time;
                latest_is_write[key] = 1;
                last_write_time[key] = actual_time;
            } else {
                if (actual_time > latest_end[key]) {
                    latest_end[key] = actual_time;
                }
                latest_is_write[key] = 0;
            }
        }
    }

    free(latest_end);
    free(latest_is_write);
    free(last_write_time);
    return total_cost;
}

// Find best insertion position - returns best_pos in out_pos[0], best_cost in out_cost[0]
void find_best_pos_c(int* partial_seq, int m, int txn, int* ops_flat, int* offsets, int* num_ops_arr, int num_keys, int* out_pos, int* out_cost) {
    // Allocate prefix states
    int state_size = num_keys;
    int* prefix_le = (int*)malloc((m + 1) * state_size * sizeof(int));
    int* prefix_liw = (int*)malloc((m + 1) * state_size * sizeof(int));
    int* prefix_lwt = (int*)malloc((m + 1) * state_size * sizeof(int));
    int* prefix_tc = (int*)malloc((m + 1) * sizeof(int));

    // Initialize prefix[0]
    memset(prefix_le, -1, state_size * sizeof(int));
    memset(prefix_liw, 0, state_size * sizeof(int));
    memset(prefix_lwt, -1, state_size * sizeof(int));
    prefix_tc[0] = 0;

    // Build prefix states
    int* le = (int*)malloc(state_size * sizeof(int));
    int* liw = (int*)malloc(state_size * sizeof(int));
    int* lwt = (int*)malloc(state_size * sizeof(int));
    memcpy(le, prefix_le, state_size * sizeof(int));
    memcpy(liw, prefix_liw, state_size * sizeof(int));
    memcpy(lwt, prefix_lwt, state_size * sizeof(int));
    int tc = 0;

    for (int idx = 0; idx < m; idx++) {
        int ti = partial_seq[idx];
        int start_op = offsets[ti] * 4;
        int nops = num_ops_arr[ti];
        int txn_start = 1;
        int txn_len = ops_flat[start_op + 3];

        for (int j = 0; j < nops; j++) {
            int base = start_op + j * 4;
            int is_write = ops_flat[base];
            int key = ops_flat[base + 1];
            int pos = ops_flat[base + 2];
            if (le[key] >= 0) {
                int constraint;
                if (liw[key] || is_write) {
                    constraint = le[key] + 2 - pos;
                } else {
                    constraint = (lwt[key] >= 0) ? (lwt[key] + 2 - pos) : (1 - pos);
                }
                if (constraint > txn_start) txn_start = constraint;
            }
        }

        int txn_end = txn_start + txn_len - 1;
        int cost_inc = txn_end - tc;
        if (cost_inc < 0) cost_inc = 0;
        tc += cost_inc;

        for (int j = 0; j < nops; j++) {
            int base = start_op + j * 4;
            int is_write = ops_flat[base];
            int key = ops_flat[base + 1];
            int pos = ops_flat[base + 2];
            int actual_time = txn_start + pos - 1;
            if (is_write) {
                le[key] = actual_time;
                liw[key] = 1;
                lwt[key] = actual_time;
            } else {
                if (actual_time > le[key]) le[key] = actual_time;
                liw[key] = 0;
            }
        }

        // Store prefix state for idx+1
        int off = (idx + 1) * state_size;
        memcpy(prefix_le + off, le, state_size * sizeof(int));
        memcpy(prefix_liw + off, liw, state_size * sizeof(int));
        memcpy(prefix_lwt + off, lwt, state_size * sizeof(int));
        prefix_tc[idx + 1] = tc;
    }

    // Now try inserting txn at each position
    int txn_start_op = offsets[txn] * 4;
    int txn_nops = num_ops_arr[txn];
    int txn_len_t = ops_flat[txn_start_op + 3];

    int best_cost = 2000000000;
    int best_pos = 0;

    for (int pos = 0; pos <= m; pos++) {
        // Load prefix state at pos
        int poff = pos * state_size;
        memcpy(le, prefix_le + poff, state_size * sizeof(int));
        memcpy(liw, prefix_liw + poff, state_size * sizeof(int));
        memcpy(lwt, prefix_lwt + poff, state_size * sizeof(int));
        tc = prefix_tc[pos];

        // Insert the new txn
        int ts = 1;
        for (int j = 0; j < txn_nops; j++) {
            int base = txn_start_op + j * 4;
            int is_write = ops_flat[base];
            int key = ops_flat[base + 1];
            int p = ops_flat[base + 2];
            if (le[key] >= 0) {
                int constraint;
                if (liw[key] || is_write) {
                    constraint = le[key] + 2 - p;
                } else {
                    constraint = (lwt[key] >= 0) ? (lwt[key] + 2 - p) : (1 - p);
                }
                if (constraint > ts) ts = constraint;
            }
        }

        int txn_end = ts + txn_len_t - 1;
        int cost_inc = txn_end - tc;
        if (cost_inc < 0) cost_inc = 0;
        tc += cost_inc;

        // Update state for the inserted txn
        for (int j = 0; j < txn_nops; j++) {
            int base = txn_start_op + j * 4;
            int is_write = ops_flat[base];
            int key = ops_flat[base + 1];
            int p = ops_flat[base + 2];
            int actual_time = ts + p - 1;
            if (is_write) {
                le[key] = actual_time;
                liw[key] = 1;
                lwt[key] = actual_time;
            } else {
                if (actual_time > le[key]) le[key] = actual_time;
                liw[key] = 0;
            }
        }

        // Continue with remaining elements
        int early_break = 0;
        for (int idx = pos; idx < m; idx++) {
            int ti = partial_seq[idx];
            int so = offsets[ti] * 4;
            int nops = num_ops_arr[ti];
            int ts2 = 1;
            int tl2 = ops_flat[so + 3];

            for (int j = 0; j < nops; j++) {
                int base = so + j * 4;
                int is_write = ops_flat[base];
                int key = ops_flat[base + 1];
                int p = ops_flat[base + 2];
                if (le[key] >= 0) {
                    int constraint;
                    if (liw[key] || is_write) {
                        constraint = le[key] + 2 - p;
                    } else {
                        constraint = (lwt[key] >= 0) ? (lwt[key] + 2 - p) : (1 - p);
                    }
                    if (constraint > ts2) ts2 = constraint;
                }
            }

            int te2 = ts2 + tl2 - 1;
            int ci2 = te2 - tc;
            if (ci2 < 0) ci2 = 0;
            tc += ci2;

            if (tc >= best_cost) {
                early_break = 1;
                break;
            }

            for (int j = 0; j < nops; j++) {
                int base = so + j * 4;
                int is_write = ops_flat[base];
                int key = ops_flat[base + 1];
                int p = ops_flat[base + 2];
                int actual_time = ts2 + p - 1;
                if (is_write) {
                    le[key] = actual_time;
                    liw[key] = 1;
                    lwt[key] = actual_time;
                } else {
                    if (actual_time > le[key]) le[key] = actual_time;
                    liw[key] = 0;
                }
            }
        }

        if (tc < best_cost) {
            best_cost = tc;
            best_pos = pos;
        }
    }

    out_pos[0] = best_pos;
    out_cost[0] = best_cost;

    free(prefix_le);
    free(prefix_liw);
    free(prefix_lwt);
    free(prefix_tc);
    free(le);
    free(liw);
    free(lwt);
}
"""

    # Compile C code
    tmpdir = tempfile.mkdtemp()
    c_path = os.path.join(tmpdir, "eval.c")
    so_path = os.path.join(tmpdir, "eval.so")
    with open(c_path, "w") as f:
        f.write(c_code)
    os.system(f"gcc -O3 -shared -fPIC -o {so_path} {c_path}")
    lib = ctypes.CDLL(so_path)

    # Setup C function signatures
    lib.fast_eval_c.restype = ctypes.c_int
    lib.fast_eval_c.argtypes = [
        ctypes.POINTER(ctypes.c_int),  # seq
        ctypes.c_int,                  # n
        ctypes.POINTER(ctypes.c_int),  # ops_flat
        ctypes.POINTER(ctypes.c_int),  # offsets
        ctypes.POINTER(ctypes.c_int),  # num_ops
        ctypes.c_int,                  # num_keys
    ]

    lib.fast_eval_with_costs_c.restype = ctypes.c_int
    lib.fast_eval_with_costs_c.argtypes = [
        ctypes.POINTER(ctypes.c_int),  # seq
        ctypes.c_int,                  # n
        ctypes.POINTER(ctypes.c_int),  # ops_flat
        ctypes.POINTER(ctypes.c_int),  # offsets
        ctypes.POINTER(ctypes.c_int),  # num_ops
        ctypes.c_int,                  # num_keys
        ctypes.POINTER(ctypes.c_int),  # cost_out
    ]

    lib.find_best_pos_c.restype = None
    lib.find_best_pos_c.argtypes = [
        ctypes.POINTER(ctypes.c_int),  # partial_seq
        ctypes.c_int,                  # m
        ctypes.c_int,                  # txn
        ctypes.POINTER(ctypes.c_int),  # ops_flat
        ctypes.POINTER(ctypes.c_int),  # offsets
        ctypes.POINTER(ctypes.c_int),  # num_ops
        ctypes.c_int,                  # num_keys
        ctypes.POINTER(ctypes.c_int),  # out_pos
        ctypes.POINTER(ctypes.c_int),  # out_cost
    ]

    # Create C arrays
    ops_c = (ctypes.c_int * (total_ops * 4))(*all_ops_flat)
    offsets_c = (ctypes.c_int * (n + 1))(*txn_offsets)
    num_ops_c = (ctypes.c_int * n)(*txn_num_ops)

    seq_c = (ctypes.c_int * n)()
    cost_out_c = (ctypes.c_int * n)()
    partial_c = (ctypes.c_int * n)()
    out_pos_c = (ctypes.c_int * 1)()
    out_cost_c = (ctypes.c_int * 1)()

    def fast_eval(seq):
        for i in range(n):
            seq_c[i] = seq[i]
        return lib.fast_eval_c(seq_c, n, ops_c, offsets_c, num_ops_c, num_keys)

    def fast_eval_with_costs(seq):
        for i in range(n):
            seq_c[i] = seq[i]
        cost = lib.fast_eval_with_costs_c(seq_c, n, ops_c, offsets_c, num_ops_c, num_keys, cost_out_c)
        costs = [cost_out_c[i] for i in range(n)]
        return cost, costs

    def find_best_insertion_pos(partial_seq, txn):
        m = len(partial_seq)
        for i in range(m):
            partial_c[i] = partial_seq[i]
        lib.find_best_pos_c(partial_c, m, txn, ops_c, offsets_c, num_ops_c, num_keys, out_pos_c, out_cost_c)
        return out_pos_c[0], out_cost_c[0]

    def neh_construction(txn_order=None):
        """NEH construction using C evaluator."""
        if txn_order is None:
            txn_order = list(range(n))
            random.shuffle(txn_order)

        seq = [txn_order[0]]
        for i in range(1, n):
            if elapsed() > time_budget * 0.95:
                seq.append(txn_order[i])
                continue
            txn = txn_order[i]
            best_pos, _ = find_best_insertion_pos(seq, txn)
            seq.insert(best_pos, txn)

        cost = fast_eval(seq)
        return cost, seq

    def neh_with_priority():
        """NEH with conflict-based priority ordering."""
        conflict_counts = []
        for i in range(n):
            txn_ops = workload.txns[i]
            writes = sum(1 for op in txn_ops if op[0] == 'w')
            total_ops = len(txn_ops)
            conflict_counts.append((writes * 2 + total_ops, i))
        conflict_counts.sort(reverse=True)
        txn_order = [t[1] for t in conflict_counts]
        return neh_construction(txn_order)

    def iterated_greedy_guided(init_schedule, init_cost, ig_time_budget, d=3):
        """IG with guided destruction targeting high-cost transactions."""
        best = list(init_schedule)
        best_cost = init_cost
        current = list(init_schedule)
        current_cost = init_cost

        ig_start = time.time()
        T_start = best_cost * 0.04
        T_end = 0.01
        iterations = 0
        cached_costs = None

        while True:
            now = time.time()
            ig_elapsed = now - ig_start
            if ig_elapsed >= ig_time_budget:
                break

            progress = ig_elapsed / ig_time_budget
            T = T_start * ((T_end / T_start) ** progress)

            # Guided destruction: pick from high-cost positions
            if iterations % 4 == 0:
                _, cached_costs = fast_eval_with_costs(current)

            if cached_costs and iterations % 2 == 0:
                # Pick from top-k highest cost positions
                indexed = list(enumerate(cached_costs))
                indexed.sort(key=lambda x: -x[1])
                # Pick from top 30% with randomness
                top_k = max(d + 2, n // 3)
                pool = [idx for idx, _ in indexed[:top_k]]
                indices = sorted(random.sample(pool, d), reverse=True)
            else:
                indices = sorted(random.sample(range(n), d), reverse=True)

            # Destruction
            candidate = list(current)
            removed_txns = []
            for idx in indices:
                removed_txns.append(candidate.pop(idx))
            random.shuffle(removed_txns)

            # Reconstruction: greedy reinsertion
            for txn in removed_txns:
                best_pos, _ = find_best_insertion_pos(candidate, txn)
                candidate.insert(best_pos, txn)

            new_cost = fast_eval(candidate)

            delta = new_cost - current_cost
            if delta < 0:
                current = candidate
                current_cost = new_cost
                cached_costs = None  # invalidate cache
                if current_cost < best_cost:
                    best_cost = current_cost
                    best = list(current)
            elif T > 0 and random.random() < math.exp(-delta / T):
                current = candidate
                current_cost = new_cost
                cached_costs = None

            # Restart from best if drifted too far
            if random.random() < 0.005 and current_cost > best_cost * 1.3:
                current = list(best)
                current_cost = best_cost
                cached_costs = None

            iterations += 1

        return best_cost, best

    def sa_phase(init_schedule, init_cost, sa_time_budget):
        """Fast SA using C evaluator."""
        best = list(init_schedule)
        best_cost = init_cost
        current = list(init_schedule)
        current_cost = init_cost

        sa_start = time.time()
        T_start = best_cost * 0.06
        T_end = 0.001
        stagnation = 0

        while True:
            now = time.time()
            sa_elapsed = now - sa_start
            if sa_elapsed >= sa_time_budget:
                break

            progress = sa_elapsed / sa_time_budget
            T = T_start * ((T_end / T_start) ** progress)

            r = random.random()
            if r < 0.5:
                # Swap
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                current[i], current[j] = current[j], current[i]
                new_cost = fast_eval(current)
                delta = new_cost - current_cost

                if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best = list(current)
                    stagnation = 0
                else:
                    current[i], current[j] = current[j], current[i]
                    stagnation += 1

            elif r < 0.85:
                # Insertion
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                txn = current.pop(i)
                if j > i:
                    j -= 1
                current.insert(j, txn)
                new_cost = fast_eval(current)
                delta = new_cost - current_cost

                if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best = list(current)
                    stagnation = 0
                else:
                    current.pop(j)
                    current.insert(i, txn)
                    stagnation += 1

            else:
                # Or-opt: move segment of 2-3
                seg_len = random.randint(2, 3)
                if seg_len >= n:
                    continue
                i = random.randint(0, n - seg_len)
                segment = current[i:i + seg_len]
                del current[i:i + seg_len]
                j = random.randint(0, len(current))
                current[j:j] = segment
                new_cost = fast_eval(current)
                delta = new_cost - current_cost

                if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best = list(current)
                    stagnation = 0
                else:
                    del current[j:j + seg_len]
                    current[i:i] = segment
                    stagnation += 1

            if stagnation > 50000:
                current = list(best)
                current_cost = best_cost
                stagnation = 0

        return best_cost, best

    # Phase 1: NEH construction (fast with C evaluator)
    best_cost = float('inf')
    best_schedule = None

    cost, sched = neh_with_priority()
    if cost < best_cost:
        best_cost = cost
        best_schedule = sched

    # Many random NEH variants (C makes this very fast)
    construction_deadline = time_budget * 0.12
    while elapsed() < construction_deadline:
        cost, sched = neh_construction()
        if cost < best_cost:
            best_cost = cost
            best_schedule = sched

    # Phase 2: IG with d=4 for aggressive exploration
    remaining_time = time_budget - elapsed()
    if remaining_time > 10:
        ig_cost, ig_sched = iterated_greedy_guided(
            best_schedule, best_cost, remaining_time * 0.35, d=4
        )
        if ig_cost < best_cost:
            best_cost = ig_cost
            best_schedule = ig_sched

    # Phase 3: IG with d=3
    remaining_time = time_budget - elapsed()
    if remaining_time > 10:
        ig_cost2, ig_sched2 = iterated_greedy_guided(
            best_schedule, best_cost, remaining_time * 0.45, d=3
        )
        if ig_cost2 < best_cost:
            best_cost = ig_cost2
            best_schedule = ig_sched2

    # Phase 4: IG with d=2 for fine-tuning
    remaining_time = time_budget - elapsed()
    if remaining_time > 10:
        ig_cost3, ig_sched3 = iterated_greedy_guided(
            best_schedule, best_cost, remaining_time * 0.45, d=2
        )
        if ig_cost3 < best_cost:
            best_cost = ig_cost3
            best_schedule = ig_sched3

    # Phase 5: SA refinement with C-accelerated evaluation
    remaining_time = time_budget - elapsed()
    if remaining_time > 5:
        sa_cost, sa_sched = sa_phase(
            best_schedule, best_cost, remaining_time * 0.90
        )
        if sa_cost < best_cost:
            best_cost = sa_cost
            best_schedule = sa_sched

    # Verify with official evaluator
    final_cost = workload.get_opt_seq_cost(best_schedule)
    return final_cost, best_schedule

# EVOLVE-BLOCK-END

def get_random_costs():
    workload_size = 100
    workload = Workload(WORKLOAD_1)

    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    cost3 = workload3.get_opt_seq_cost(schedule3)
    print(cost1, cost2, cost3)
    return cost1 + cost2 + cost3, [schedule1, schedule2, schedule3]


if __name__ == "__main__":
    makespan, schedule = get_random_costs()
    print(f"Makespan: {makespan}")

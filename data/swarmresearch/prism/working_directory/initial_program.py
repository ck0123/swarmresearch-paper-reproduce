GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR using binary search + multi-start greedy + iterated local search.
    """
    import random

    if not models:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    n = len(models)

    def get_max_kvpr(gpu_weight, gpu_size):
        max_k = 0.0
        for i in range(gpu_num):
            if gpu_size[i] > 0:
                rem = GPU_MEM_SIZE - gpu_size[i]
                if rem <= 0:
                    return float('inf')
                k = gpu_weight[i] / rem
                if k > max_k:
                    max_k = k
        return max_k

    def try_binpack(T):
        """Bin-packing feasibility check with BFD."""
        cap = T * GPU_MEM_SIZE
        items = [(m.req_rate / m.slo + T * m.model_size, m) for m in models]
        items.sort(key=lambda x: x[0], reverse=True)
        bins = [0.0] * gpu_num
        bin_sizes = [0.0] * gpu_num
        assignment = [0] * len(items)
        for idx, (weight, model) in enumerate(items):
            best_g = -1
            best_remaining = float('inf')
            for g in range(gpu_num):
                rem = cap - bins[g]
                if weight <= rem + 1e-12 and bin_sizes[g] + model.model_size <= GPU_MEM_SIZE:
                    if rem < best_remaining:
                        best_remaining = rem
                        best_g = g
            if best_g < 0:
                for g in range(gpu_num):
                    rem = cap - bins[g]
                    if weight <= rem + 1e-12 and bin_sizes[g] + model.model_size <= GPU_MEM_SIZE:
                        best_g = g
                        break
            if best_g < 0:
                return None
            bins[best_g] += weight
            bin_sizes[best_g] += model.model_size
            assignment[idx] = best_g
        placement = {g: [] for g in range(gpu_num)}
        gw = [0.0] * gpu_num
        gs = [0.0] * gpu_num
        for idx, (_, model) in enumerate(items):
            g = assignment[idx]
            placement[g].append(model)
            gw[g] += model.req_rate / model.slo
            gs[g] += model.model_size
        return placement, gw, gs

    def lookahead_greedy(sorted_models):
        placement = {g: [] for g in range(gpu_num)}
        gw = [0.0] * gpu_num
        gs = [0.0] * gpu_num
        for model in sorted_models:
            mw = model.req_rate / model.slo
            ms = model.model_size
            best_idx = -1
            best_kvpr = float('inf')
            for g in range(gpu_num):
                new_size = gs[g] + ms
                if new_size > GPU_MEM_SIZE:
                    continue
                new_rem = GPU_MEM_SIZE - new_size
                if new_rem <= 0:
                    continue
                resulting = (gw[g] + mw) / new_rem
                if resulting < best_kvpr:
                    best_kvpr = resulting
                    best_idx = g
            if best_idx < 0:
                # Fallback: least loaded GPU
                best_idx = min(range(gpu_num), key=lambda g: gs[g])
            placement[best_idx].append(model)
            gw[best_idx] += mw
            gs[best_idx] += ms
        return placement, gw, gs

    def local_search(placement, gw, gs, max_iters=300):
        """Move and swap local search targeting the worst GPU."""
        for _ in range(max_iters):
            # Find worst GPU
            max_kvpr = -1.0
            worst_g = -1
            for g in range(gpu_num):
                if gs[g] > 0:
                    rem = GPU_MEM_SIZE - gs[g]
                    if rem <= 0:
                        return placement, gw, gs
                    k = gw[g] / rem
                    if k > max_kvpr:
                        max_kvpr = k
                        worst_g = g
            if worst_g == -1 or max_kvpr <= 0:
                break

            # Try moves from worst_g
            best_move = None
            best_new_max = max_kvpr
            for i, model in enumerate(placement[worst_g]):
                mw = model.req_rate / model.slo
                ms = model.model_size
                rem_worst = GPU_MEM_SIZE - (gs[worst_g] - ms)
                worst_after = (gw[worst_g] - mw) / rem_worst if (gw[worst_g] - mw) > 0 and rem_worst > 0 else 0.0

                for g in range(gpu_num):
                    if g == worst_g:
                        continue
                    if gs[g] + ms > GPU_MEM_SIZE:
                        continue
                    rem_g = GPU_MEM_SIZE - (gs[g] + ms)
                    if rem_g <= 0:
                        continue
                    g_after = (gw[g] + mw) / rem_g
                    new_max = max(worst_after, g_after)
                    if new_max >= best_new_max:
                        continue
                    # Check other GPUs
                    for g2 in range(gpu_num):
                        if g2 == worst_g or g2 == g:
                            continue
                        if gs[g2] > 0:
                            k2 = gw[g2] / (GPU_MEM_SIZE - gs[g2])
                            if k2 > new_max:
                                new_max = k2
                                if new_max >= best_new_max:
                                    break
                    if new_max < best_new_max - 1e-15:
                        best_new_max = new_max
                        best_move = (i, g)

            if best_move is not None:
                i, tgt = best_move
                model = placement[worst_g].pop(i)
                placement[tgt].append(model)
                mw = model.req_rate / model.slo
                ms = model.model_size
                gw[worst_g] -= mw
                gs[worst_g] -= ms
                gw[tgt] += mw
                gs[tgt] += ms
                continue

            # Try swaps from worst_g
            best_swap = None
            for i, ma in enumerate(placement[worst_g]):
                wa, sa = ma.req_rate / ma.slo, ma.model_size
                for g in range(gpu_num):
                    if g == worst_g:
                        continue
                    for j, mb in enumerate(placement[g]):
                        wb, sb = mb.req_rate / mb.slo, mb.model_size
                        new_worst_size = gs[worst_g] - sa + sb
                        new_g_size = gs[g] - sb + sa
                        if new_worst_size > GPU_MEM_SIZE or new_g_size > GPU_MEM_SIZE:
                            continue
                        rem_worst = GPU_MEM_SIZE - new_worst_size
                        rem_g = GPU_MEM_SIZE - new_g_size
                        if rem_worst <= 0 or rem_g <= 0:
                            continue
                        nw_worst = gw[worst_g] - wa + wb
                        nw_g = gw[g] - wb + wa
                        worst_after = nw_worst / rem_worst if nw_worst > 0 else 0.0
                        g_after = nw_g / rem_g if nw_g > 0 else 0.0
                        new_max = max(worst_after, g_after)
                        if new_max >= best_new_max:
                            continue
                        for g2 in range(gpu_num):
                            if g2 == worst_g or g2 == g:
                                continue
                            if gs[g2] > 0:
                                k2 = gw[g2] / (GPU_MEM_SIZE - gs[g2])
                                if k2 > new_max:
                                    new_max = k2
                                    if new_max >= best_new_max:
                                        break
                        if new_max < best_new_max - 1e-15:
                            best_new_max = new_max
                            best_swap = (i, g, j)

            if best_swap is not None:
                i, g, j = best_swap
                ma = placement[worst_g][i]
                mb = placement[g][j]
                wa, sa = ma.req_rate / ma.slo, ma.model_size
                wb, sb = mb.req_rate / mb.slo, mb.model_size
                placement[worst_g][i] = mb
                placement[g][j] = ma
                gw[worst_g] += wb - wa
                gs[worst_g] += sb - sa
                gw[g] += wa - wb
                gs[g] += sa - sb
                continue

            break  # No improvement found

        return placement, gw, gs

    best_placement = None
    best_max_kvpr = float('inf')
    rng = random.Random(42)

    def update_best(p, gw, gs):
        nonlocal best_placement, best_max_kvpr
        mk = get_max_kvpr(gw, gs)
        if mk < best_max_kvpr:
            best_max_kvpr = mk
            best_placement = {k: list(v) for k, v in p.items()}

    # Phase 1: Binary search on target KVPR
    total_wrr = sum(m.req_rate / m.slo for m in models)
    total_size = sum(m.model_size for m in models)
    lb = total_wrr / (gpu_num * GPU_MEM_SIZE - total_size) if gpu_num * GPU_MEM_SIZE > total_size else 0.001
    ub = max(total_wrr / max(GPU_MEM_SIZE - total_size, 0.1), lb * 20, 1.0)

    for _ in range(50):
        mid = (lb + ub) / 2
        result = try_binpack(mid)
        if result is not None:
            p, gw, gs = result
            p, gw, gs = local_search(p, gw, gs)
            update_best(p, gw, gs)
            ub = mid
        else:
            lb = mid

    # Phase 2: Multiple greedy orderings with local search
    orderings = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / max(m.model_size, 0.1), reverse=True),
        sorted(models, key=lambda m: m.model_size * (m.req_rate / m.slo), reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo + 0.5 * m.model_size, reverse=True),
    ]
    for order in orderings:
        p, gw, gs = lookahead_greedy(order)
        p, gw, gs = local_search(p, gw, gs)
        update_best(p, gw, gs)

    # Phase 3: Random restarts
    for _ in range(100):
        shuffled = list(models)
        rng.shuffle(shuffled)
        p, gw, gs = lookahead_greedy(shuffled)
        p, gw, gs = local_search(p, gw, gs)
        update_best(p, gw, gs)

    return best_placement

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Test the algorithm

    from evaluator import generate_test_gpu_models
    from evaluator import calculate_kvcache_pressure
    from evaluator import safe_float
    import numpy as np

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for i, (gpu_num, gpu_models) in enumerate(test_cases):

        results = compute_model_placement(gpu_num, gpu_models)
        max_kvpr = calculate_kvcache_pressure(results)
        all_kvpr.append(safe_float(max_kvpr))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr


    print(f"Max KVPR: {avg_kvpr:.3f}")

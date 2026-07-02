import random


# EVOLVE-BLOCK-START
def get_best_schedule(workload, num_seqs):
    """Return (makespan, schedule) for a transaction workload."""

    def get_greedy_cost_sampled(num_samples, sample_rate):
        start_txn = random.randint(0, workload.num_txns - 1)
        txn_seq = [start_txn]
        remaining_txns = list(range(workload.num_txns))
        remaining_txns.remove(start_txn)

        for _ in range(workload.num_txns - 1):
            min_cost = 100000
            min_txn = -1
            holdout_txns = []
            done = False

            if random.random() > sample_rate:
                idx = random.randint(0, len(remaining_txns) - 1)
                t = remaining_txns[idx]
                txn_seq.append(t)
                remaining_txns.pop(idx)
                continue

            for _ in range(num_samples):
                idx = 0
                if len(remaining_txns) > 1:
                    idx = random.randint(0, len(remaining_txns) - 1)
                else:
                    done = True
                t = remaining_txns[idx]
                holdout_txns.append(remaining_txns.pop(idx))
                test_seq = txn_seq.copy()
                test_seq.append(t)
                cost = workload.get_opt_seq_cost(test_seq)
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t
                if done:
                    break

            assert min_txn != -1
            txn_seq.append(min_txn)
            holdout_txns.remove(min_txn)
            remaining_txns.extend(holdout_txns)

        assert len(set(txn_seq)) == workload.num_txns
        overall_cost = workload.get_opt_seq_cost(txn_seq)
        return overall_cost, txn_seq

    return get_greedy_cost_sampled(10, 1.0)


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    from sample_workloads import SAMPLE_WORKLOADS
    from txn_simulator import Workload

    total = 0
    for raw_workload in SAMPLE_WORKLOADS:
        workload = Workload(raw_workload)
        _, schedule = get_best_schedule(workload, 10)
        total += workload.get_opt_seq_cost(schedule)
    print(total)

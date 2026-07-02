# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List
import heapq
from itertools import combinations


def search_algorithm(src, dsts, G, num_partitions):
    """
    Exact minimum directed Steiner arborescence using Dreyfus-Wagner DP.
    Cost model: data_vol * sum(unique edge costs in broadcast tree).
    With k=6-7 terminals and ~70 nodes, DP is fast: O(3^k * n + 2^k * n^2).
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    # Remove edges with None cost
    none_cost_edges = [(u, v) for u, v, d in h.edges(data=True) if d.get('cost') is None]
    h.remove_edges_from(none_cost_edges)

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Filter to reachable destinations
    reachable_dsts = []
    for dst in dsts:
        if nx.has_path(h, src, dst):
            reachable_dsts.append(dst)

    if not reachable_dsts:
        return bc_topology

    # Get all nodes
    nodes = list(h.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    k = len(reachable_dsts)

    # Compute shortest path costs from every node to every destination
    # Using reverse Dijkstra: shortest path in reversed graph from dest
    h_rev = h.reverse()

    # sp_to_dst[d_idx][node_idx] = shortest path cost from node to destination d
    sp_to_dst = {}
    sp_path_to_dst = {}
    for d_idx, dst in enumerate(reachable_dsts):
        lengths = nx.single_source_dijkstra_path_length(h_rev, dst, weight='cost')
        sp_to_dst[d_idx] = {node_to_idx[n_name]: c for n_name, c in lengths.items() if n_name in node_to_idx}
        paths = nx.single_source_dijkstra_path(h_rev, dst, weight='cost')
        sp_path_to_dst[d_idx] = {}
        for n_name, rev_path in paths.items():
            if n_name in node_to_idx:
                sp_path_to_dst[d_idx][node_to_idx[n_name]] = list(reversed(rev_path))

    # Dreyfus-Wagner DP for minimum Steiner tree
    # dp[S][v] = minimum cost of subtree rooted at v reaching all terminals in S
    INF = float('inf')
    full_mask = (1 << k) - 1

    dp = [{} for _ in range(1 << k)]
    parent_info = [{} for _ in range(1 << k)]

    # Base cases: singleton sets
    for d_idx in range(k):
        mask = 1 << d_idx
        for v_idx, cost in sp_to_dst[d_idx].items():
            dp[mask][v_idx] = cost

    # Fill DP in order of increasing subset size
    for s in range(1, full_mask + 1):
        if bin(s).count('1') < 2:
            continue

        # Combine: try all non-trivial partitions of s
        sub = (s - 1) & s
        while sub > 0:
            comp = s ^ sub
            if sub < comp:  # avoid double counting
                for v_idx in range(n):
                    c1 = dp[sub].get(v_idx, INF)
                    c2 = dp[comp].get(v_idx, INF)
                    if c1 + c2 < dp[s].get(v_idx, INF):
                        dp[s][v_idx] = c1 + c2
                        parent_info[s][v_idx] = ('split', sub, comp)
            sub = (sub - 1) & s

        # Relax using Dijkstra on edges (propagate through graph)
        # dp[s][u] = min(dp[s][u], dp[s][v] + cost(u,v)) for edge (u,v)
        pq = []
        for v_idx, cost in dp[s].items():
            heapq.heappush(pq, (cost, v_idx))

        visited = set()
        while pq:
            cost, v_idx = heapq.heappop(pq)
            if v_idx in visited:
                continue
            if cost > dp[s].get(v_idx, INF):
                continue
            visited.add(v_idx)

            v = nodes[v_idx]
            # For edge (u, v) in h: dp[s][u] = dp[s][v] + cost(u,v)
            for u in h.predecessors(v):
                u_idx = node_to_idx[u]
                edge_cost = h[u][v]['cost']
                new_cost = cost + edge_cost
                if new_cost < dp[s].get(u_idx, INF):
                    dp[s][u_idx] = new_cost
                    parent_info[s][u_idx] = ('edge', v_idx, s)
                    heapq.heappush(pq, (new_cost, u_idx))

    # Answer is dp[full_mask][src_idx]
    src_idx = node_to_idx[src]

    if dp[full_mask].get(src_idx, INF) == INF:
        # Fallback to independent shortest paths
        for dst in reachable_dsts:
            path = nx.dijkstra_path(h, src, dst, weight='cost')
            for i in range(len(path) - 1):
                s_node, t_node = path[i], path[i + 1]
                for j in range(num_partitions):
                    bc_topology.append_dst_partition_path(dst, j, [s_node, t_node, G[s_node][t_node]])
        return bc_topology

    # Reconstruct the Steiner tree
    tree_edges = set()
    _reconstruct_tree(src_idx, full_mask, parent_info, dp, sp_path_to_dst, nodes, node_to_idx, h, tree_edges, k)

    # Build paths from tree edges using BFS/DFS from src
    tree_graph = nx.DiGraph()
    for u, v in tree_edges:
        tree_graph.add_edge(u, v)

    # Find path from src to each destination in the tree
    for dst in reachable_dsts:
        if tree_graph.has_node(dst) and nx.has_path(tree_graph, src, dst):
            path = nx.shortest_path(tree_graph, src, dst)
        else:
            # Fallback: direct shortest path
            path = nx.dijkstra_path(h, src, dst, weight='cost')

        for i in range(len(path) - 1):
            s_node, t_node = path[i], path[i + 1]
            for j in range(num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s_node, t_node, G[s_node][t_node]])

    return bc_topology


def _reconstruct_tree(v_idx, mask, parent_info, dp, sp_path_to_dst, nodes, node_to_idx, h, tree_edges, k):
    """Recursively reconstruct the Steiner tree from DP parent pointers."""
    if mask == 0:
        return

    # Singleton - use direct shortest path to that terminal
    if bin(mask).count('1') == 1:
        d_idx = 0
        temp = mask
        while temp > 1:
            d_idx += 1
            temp >>= 1

        if v_idx in sp_path_to_dst[d_idx]:
            path = sp_path_to_dst[d_idx][v_idx]
            for i in range(len(path) - 1):
                tree_edges.add((path[i], path[i + 1]))
        return

    if v_idx not in parent_info[mask]:
        # No parent info - fall back to individual paths
        for d_idx in range(k):
            if mask & (1 << d_idx):
                if v_idx in sp_path_to_dst[d_idx]:
                    path = sp_path_to_dst[d_idx][v_idx]
                    for i in range(len(path) - 1):
                        tree_edges.add((path[i], path[i + 1]))
        return

    info = parent_info[mask][v_idx]

    if info[0] == 'split':
        _, s1, s2 = info
        _reconstruct_tree(v_idx, s1, parent_info, dp, sp_path_to_dst, nodes, node_to_idx, h, tree_edges, k)
        _reconstruct_tree(v_idx, s2, parent_info, dp, sp_path_to_dst, nodes, node_to_idx, h, tree_edges, k)
    elif info[0] == 'edge':
        _, next_v_idx, next_mask = info
        v_name = nodes[v_idx]
        next_v_name = nodes[next_v_idx]
        tree_edges.add((v_name, next_v_name))
        _reconstruct_tree(next_v_idx, next_mask, parent_info, dp, sp_path_to_dst, nodes, node_to_idx, h, tree_edges, k)


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src
        self.dsts = dsts
        self.num_partitions = num_partitions

        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        print(f"now the set path is: {self.paths}")
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)

def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if cost_path is None:
        cost = pd.read_csv(os.path.join(current_dir, "profiles/cost.csv"))
    else:
        cost = pd.read_csv(cost_path)

    if throughput_path is None:
        throughput = pd.read_csv(os.path.join(current_dir, "profiles/throughput.csv"))
    else:
        throughput = pd.read_csv(throughput_path)

    G = nx.DiGraph()
    for _, row in throughput.iterrows():
        if row["src_region"] == row["dst_region"]:
            continue
        G.add_edge(row["src_region"], row["dst_region"], cost=None, throughput=num_vms * row["throughput_sent"] / 1e9)

    for _, row in cost.iterrows():
        if row["src"] in G and row["dest"] in G[row["src"]]:
            G[row["src"]][row["dest"]]["cost"] = row["cost"]

    no_cost_pairs = []
    for edge in G.edges.data():
        src, dst = edge[0], edge[1]
        if edge[-1]["cost"] is None:
            no_cost_pairs.append((src, dst))
    print("Unable to get costs for: ", no_cost_pairs)

    return G


# EVOLVE-BLOCK-END

# Helper functions that won't be evolved
def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance"""
    return BroadCastTopology(src, dsts, num_partitions)

def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)

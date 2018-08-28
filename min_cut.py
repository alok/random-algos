#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""An implementation of Karger's algorithm."""

import random
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np


AdjacencyMatrix = np.ndarray
Vertex = int
Edge = Tuple[Vertex, Vertex]  # of len = 2
AdjacencyList = Dict[Vertex, List[Vertex]]


def gen_sym(V: int = 10) -> np.ndarray:
    G = np.triu(np.random.randint(0, 1 + 1, size=(V, V)))
    sym = G + G.T
    # rm self-edges
    np.fill_diagonal(sym, 0)
    return sym


def gen_adj_list(G: AdjacencyMatrix) -> AdjacencyList:
    # {v : vertices that have a 1 in the adjacency matrix}
    return {v: list(np.argwhere(vertices).reshape(-1)) for v, vertices in enumerate(G)}


def gen_all_edges(A: AdjacencyList) -> List[Edge]:

    # avoid adding both (v,w) and (w,v)
    edges = []
    for v in A:
        for i in A[v]:
            if (i, v) not in edges:
                edges.append((v, i))
    return edges


def select_edge(A: AdjacencyList) -> Edge:
    return random.choice(gen_all_edges(A))


def replace_value(xs: List, value, replacement):
    return [replacement if x == value else x for x in xs]


def contract_edge(edge: Edge, A: AdjacencyList) -> None:

    v, w = edge

    # rm *all* edges between v and w
    A[v] = [vertex for vertex in A[v] if vertex != w]
    A[w] = [vertex for vertex in A[w] if vertex != v]

    # Add w's edges to v. No self edges to worry about by
    # assumption/construction above.
    A[v].extend(A[w])

    # rm w since it's collapsed into v
    del A[w]

    # We don't use a dict comprehension since we want to mutate A in-place
    for vertex, neighbors in A.items():
        A[vertex] = replace_value(neighbors, w, v)


def test_single(A: AdjacencyList, num_iters: int = 10) -> int:
    return min(min_cut_len(deepcopy(A)) for _ in range(num_iters))


def min_cut_len(A: AdjacencyList) -> int:
    while len(A) > 2:
        contract_edge(select_edge(A), A)

    return len(list(A.values())[0])


if __name__ == "__main__":
    G = gen_sym(V=10)
    A = gen_adj_list(G)

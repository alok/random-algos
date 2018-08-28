#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np


AdjacencyMatrix = np.ndarray
Vertex = int
Edge = Tuple[Vertex, Vertex]  # of len = 2
AdjacencyList = Dict[Vertex, List[Vertex]]


# TODO generate symmetric matrix properly
def gen_sym(V: int = 10) -> np.ndarray:
    G = np.random.randint(0, 1 + 1, size=(V, V))
    G += G.T
    G[G > 1] = 1
    np.fill_diagonal(G,0)
    return G


def gen_adj_list(G: AdjacencyMatrix) -> AdjacencyList:
    adj = {i: [] for i, _ in enumerate(G)}
    for v, row in enumerate(G):
        for c, vertex in enumerate(row):
            if row[c] == 1:
                adj[v].append(c)
    return adj




def select_edge(A: AdjacencyList) -> Edge:
    edges = [(v, i) for v in A for i in A[v]]
    return random.choice(edges)


def contract_edge(edge: Edge, A: AdjacencyList) -> None:
    # add w's connections to v

    # # rm w from all connections
    # for vertex in A:
    #     while w in A[vertex]:
    #         A[vertex].remove(w)

    # # A[v].extend([u for u in A[w] if u != v])
    # A[v].extend([u for u in A[w] if u != v])
    # del A[w]

    v, w = edge

    # rm *all* edges between v and w
    while w in A[v]:
        A[v].remove(w)
    while v in A[w]:
        A[w].remove(v)

    # No self edges to worry about by assumption/construction above.
    A[v].extend(A[w])

    del A[w]

    #
    for vertex in A:
        for i, u in enumerate(A[vertex]):
            if u == w:
                A[vertex][i] = v


def test_single(A: AdjacencyList, num_iters: int = 10) -> int:
    return min(min_cut_len(deepcopy(A)) for _ in range(num_iters))


def min_cut_len(A: AdjacencyList) -> int:
    while len(A) > 2:
        contract_edge(select_edge(A), A)

    return len(list(A.values())[0])


if __name__ == "__main__":
    G = gen_sym(V=10)
    A = gen_adj_list(G)

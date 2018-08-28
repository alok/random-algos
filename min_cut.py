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
    for i, _ in enumerate(G):
        G[i][i] = 0
    return G




def gen_adj_list(G: AdjacencyMatrix) -> AdjacencyList:
    adj = {i: set() for i, _ in enumerate(G)}
    for v, row in enumerate(G):
        for c, vertex in enumerate(row):
            if row[c] == 1:
                adj[v].add(c)
    return adj


Edge = Tuple[Vertex, Vertex]  # of len = 2


def select_edge(A: AdjacencyList) -> Edge:
    edges = [(v, i) for v in A for i in A[v]]
    return random.choice(edges)


# TODO rm deepcopy and do this in place
def contract_edge(edge: Edge, A: AdjacencyList) -> AdjacencyList:
    v, w = edge
    A = deepcopy(A)
    # add w's connections to v
    A[v].update(A[w])
    del A[w]
    # rm w from all connections
    for vertex in A:
        A[vertex].discard(w)

    return A

if __name__ == "__main__":
    G = gen_sym(V=10)
    A = gen_adj_list(G)
    contract_edge(select_edge(A), A)


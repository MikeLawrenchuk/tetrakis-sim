import networkx as nx
from itertools import product

# Build one slice ------------------------------------------
V = [(r, c, q) for r in range(9) for c in range(9) for q in 'ABCD']
G = nx.Graph(); G.add_nodes_from(V)

def clique(nodes):                    # helper
    for i, v in enumerate(nodes):
        for w in nodes[i+1:]:
            G.add_edge(v, w)

# stock constraints (flat sheet)
for r, c in product(range(9), repeat=2):
    clique([(r, c, q) for q in 'ABCD'])          # internal
for r in range(9):
    for q in 'ABCD':
        clique([(r, c, q) for c in range(9)])    # rows
for c in range(9):
    for q in 'ABCD':
        clique([(r, c, q) for r in range(9)])    # columns

# -----------------------------------------------------------
# "insert mass" at the centre by *removing* one internal edge
# (equivalent to dropping one triangle)
G.remove_edge((4,4,'A'), (4,4,'B'))   # 45° wedge removed
# inspect curvature: degree of those two vertices ↓ by 1
print(G.degree((4,4,'A')), G.degree((4,4,'B')))

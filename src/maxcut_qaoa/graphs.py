import networkx as nx
import json
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt


SEED = 42

def make_cubic(n, seed):
    """
    Create a 3-regular graph with n nodes
        - n: nodes in the graph. Must be even and > 6
        - seed: used to generate the graph for reproducibility
    """
    
    return nx.random_regular_graph(3, n, seed=seed)

sizes = [8, 10, 12, 14, 18, 20, 24, 30, 36]

graphs = {}

for n in sizes:
    G_reg = make_cubic(n, seed=100 + n)
    graphs[n] = G_reg ## add the regular graph to the dictionary
    
for n, graph in graphs.items():
    pos = nx.spring_layout(graph, seed=SEED)
    nx.draw(graph, pos=pos, with_labels=True,
             edge_color="gray", node_color="lightblue",
             node_size=400, font_size=10)
    plt.title(f"n={n}")
    plt.show()        
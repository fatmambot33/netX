from collections.abc import Iterable
import logging
import types
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Any
from heapq import nlargest
import matplotlib.pyplot as plt
import networkx as nx
from networkx.classes.graph import Graph
import numpy as np
import pandas as pd
from networkx import Graph
from networkx.algorithms import approximation
from pandas.api.types import is_numeric_dtype

# Hint for Visual Code Python Interactive window
# %%


DEFAULT = {"MAX_EDGES": 100,
           "MAX_NODES": 50,
           "MAX_NODE_SIZE": 600,
           "MAX_EDGE_WIDTH": 10,
           "GRAPH_SCALE": 2
           }


class NodeView(nx.classes.reportviews.NodeView):
    def sort(self,
             attribute: Optional[str] = 'weight',
             reverse: Optional[bool] = True):
        # Sort the nodes based on the specified attribute
        sorted_nodes = sorted(self,
                              key=lambda node: self[node][attribute],
                              reverse=reverse)
        return sorted_nodes

    def filter(self, attribute: str, value: str):
        # Filter the nodes based on the specified attribute and value
        filtered_nodes = [
            node for node in self if attribute in self[node] and self[node][attribute] == value]
        return filtered_nodes


class AdjacencyView(nx.classes.coreviews.AdjacencyView):
    def sort(self,
             attribute: Optional[str] = 'weight',
             reverse: Optional[bool] = True):
        # Sort the nodes based on the specified attribute
        sorted_nodes = sorted(self,
                              key=lambda node: self[node][attribute],
                              reverse=reverse)
        return sorted_nodes

    def filter(self, attribute: str, value: str):
        # Filter the nodes based on the specified attribute and value
        filtered_nodes = [
            node for node in self if attribute in self[node] and self[node][attribute] == value]
        return filtered_nodes


class EdgeView(nx.classes.reportviews.EdgeView):
    def sort(self,
             reverse: Optional[bool] = True,
             attribute: Optional[str] = 'weight'):
        sorted_edges = sorted(self(data=True),
                              key=lambda t: t[2].get(attribute, 1),
                              reverse=reverse)
        return {(u, v): _ for u, v, _ in sorted_edges}

    def filter(self, attribute: str, value: str):
        # Filter the edges based on the specified attribute and value
        filtered_edges = [
            edge for edge in self if attribute in self[edge] and self[edge][attribute] == value]
        return [(edge[0], edge[1]) for edge in filtered_edges]


class Graph(nx.Graph):
    """
    Custom graph class based on NetworkX's Graph class.
    """

    def __init__(self):
        super().__init__()
        self._scale = 1.0

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float):
        self._scale = value

    @property
    def nodes(self):
        return NodeView(self)

    @property
    def edges(self):
        return EdgeView(self)

    @property
    def adjacency(self):
        return AdjacencyView(list(self))


    def layout(self, max_node_size: int = DEFAULT["MAX_NODES"], max_edge_width: int = DEFAULT["MAX_EDGE_WIDTH"], max_font_size: int = 14):
        """
        Calculates the sizes for nodes, edges, and fonts based on node weights and edge weights.

        Parameters:
        - max_node_size (int): Maximum size for nodes (default: 300).
        - max_edge_width (int): Maximum width for edges (default: 10).
        - max_font_size (int): Maximum font size for node labels (default: 18).

        Returns:
        - Tuple[List[int], List[int], Dict[int, List[str]]]: A tuple containing the node sizes, edge widths,
          and font sizes for node labels.
        """

        node_weights = [data.get('weight_normalized', 1)
                        for node, data in self.nodes(data=True)]
        node_size = [weight*max_node_size for weight in node_weights]

        # Normalize edge weights between 0 and 1
        edge_weights = [data.get('weight', 0)
                        for _, _, data in self.edges(data=True)]
        min_edge_weight = min(edge_weights)
        max_edge_weight = max(edge_weights)
        if max_edge_weight - min_edge_weight == 0:
            normalized_edge_weights = [0 for weight in edge_weights]
        else:
            normalized_edge_weights = [
                (weight - min_edge_weight) / (max_edge_weight - min_edge_weight) for weight in edge_weights]

        # Scale the normalized edge weights within the desired range of edge widths
        edges_width = [
            width * max_edge_width for width in normalized_edge_weights]

        # Scale the normalized node weights within the desired range of font sizes
        node_size_dict = dict(zip(self.nodes, node_weights))
        fonts_size = defaultdict(list)
        for node, width in node_size_dict.items():
            fonts_size[int(width * max_font_size) + 6].append(node)
        fonts_size = dict(fonts_size)

        return node_size, edges_width, fonts_size

    def subgraphX(self, node_list=None, max_edges: int = DEFAULT["MAX_EDGES"]):
        if node_list is None:
            node_list = self.nodes.sort("weight")[:DEFAULT["MAX_NODES"]]
        subgraph = nx.subgraph(
            self, nbunch=node_list)
        edges = subgraph.top_k_edges(attribute="weight", k=5).keys()
        subgraph = subgraph.edge_subgraph(list(edges)[:max_edges])
        return subgraph

    def plotX(self):
        """
        Plots the degree distribution of the graph, including a degree rank plot and a degree histogram.
        """
        degree_sequence = sorted([d for n, d in self.degree()], reverse=True)
        dmax = max(degree_sequence)

        fig = plt.figure("Degree of a random graph", figsize=(8, 8))
        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(5, 4)

        ax0 = fig.add_subplot(axgrid[0:3, :])
        node_sizes, edge_widths, font_sizes = self.layout(
            DEFAULT["MAX_NODE_SIZE"], DEFAULT["MAX_EDGE_WIDTH"], 14)
        pos = nx.spring_layout(self, seed=10396953)
        # nodes
        nx.draw_networkx_nodes(self,
                               pos,
                               ax=ax0,
                               node_size=list(node_sizes),
                               # node_color=list(node_sizes.values()),
                               cmap=plt.cm.Blues)
        # edges
        nx.draw_networkx_edges(self,
                               pos,
                               ax=ax0,
                               alpha=0.4,
                               width=edge_widths)
        # labels
        for font_size, nodes in font_sizes.items():
            nx.draw_networkx_labels(
                self,
                pos,
                ax=ax0,
                font_size=font_size,
                labels={n: n for n in nodes},
                alpha=0.4)

        ax0.set_title("Connected components of G")
        ax0.set_axis_off()

        ax1 = fig.add_subplot(axgrid[3:, :2])
        ax1.plot(degree_sequence, "b-", marker="o")
        ax1.set_title("Degree Rank Plot")
        ax1.set_ylabel("Degree")
        ax1.set_xlabel("Rank")

        ax2 = fig.add_subplot(axgrid[3:, 2:])
        ax2.bar(*np.unique(degree_sequence, return_counts=True))
        ax2.set_title("Degree histogram")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("# of Nodes")

        fig.tight_layout()
        plt.show()

    def analysis(self, node_list: Optional[List] = None,
                 scale: int = DEFAULT["GRAPH_SCALE"],
                 node_scale: int = DEFAULT["MAX_NODE_SIZE"],
                 edge_scale: float = DEFAULT["MAX_EDGE_WIDTH"],
                 max_nodes: int = DEFAULT["MAX_NODES"],
                 max_edges: int = DEFAULT["MAX_EDGES"],
                 plt_title: Optional[str] = "Top keywords"):
        # node_list=self.nodes_circuits(node_list)
        g = self.subgraphX(max_edges=max_edges, node_list=node_list)
        connected_components = nx.connected_components(g)
        for connected_component in connected_components:
            connected_component_graph = self.subgraphX(max_edges=max_edges,
                                                       node_list=connected_component)
            connected_component_graph.plotX()

    def nodes_circuits(self, node_list: List[str] = [], iterations: int = 0) -> List[str]:
        """
        Finds nodes with more than one edge in a graph, by iteratively removing nodes with a single edge.

        Parameters:
        - nx_graph (Graph): The graph to search for nodes.
        - node_list (List[str]): The list of nodes to start the search (default: empty list).
        - iterations (int): The number of iterations performed (default: 0).

        Returns:
        - List[str]: The list of nodes with more than one edge.
        """
        if not node_list:
            node_list = list(self)

        single_edge_nodes = [
            node for node in node_list if self.degree(node) <= 1]

        if not single_edge_nodes:
            return node_list

        self.remove_nodes_from(single_edge_nodes)
        iterations += 1

        return Graph.nodes_circuits(self, [], iterations)

    def edge_subgraph(self, edges: Iterable) -> Graph:
        return nx.edge_subgraph(self, edges)

    def top_k_edges(self, attribute: str, reverse: bool = True, k: int = 5) -> Dict[Any, List[Tuple[Any, Dict]]]:
        """
        Returns the top k edges per node based on the given attribute.

        Parameters:
        attribute (str): The attribute name to be used for sorting.
        reverse (bool): Flag indicating whether to sort in reverse order (default: True).
        k (int): Number of top edges to return per node.

        Returns:
        Dict[Any, List[Tuple[Any, Dict]]]: A dictionary where the key is a node
        and the value is a list of top k edges for that node. Each edge is represented
        as a tuple where the first element is the adjacent node and the second element
        is a dictionary of edge attributes.
        """
        top_list = {}
        for node in self.nodes:
            edges = self.edges(node, data=True)
            edges_sorted = sorted(edges, key=lambda x: x[2].get(attribute, 0), reverse=reverse)
            top_k_edges = edges_sorted[:k]
            for u, v, data in top_k_edges:
                edge_key = (u, v)
                top_list[edge_key] = data[attribute]
        return top_list

    @staticmethod
    def from_pandas_edgelist(df,
                             source: Optional[str] = "source",
                             target: Optional[str] = "target",
                             weight: Optional[str] = "weight"):
        """
        Initialize netX instance with a simple dataframe

        :param df_source: DataFrame containing network data.
        :param source: Name of source nodes column in df_source.
        :param target: Name of target nodes column in df_source.
        :param weight: Name of edges weight column in df_source.

        """
        G = Graph()
        G = nx.from_pandas_edgelist(
            df, source=source, target=target, edge_attr=weight, create_using=G)
        G.nodes_circuits()

        edge_aggregates = G.top_k_edges(attribute=weight, k=10)
        node_aggregates = {}
        for (u, v), weight_value in edge_aggregates.items():
            if u not in node_aggregates:
                node_aggregates[u] = 0
            if v not in node_aggregates:
                node_aggregates[v] = 0
            node_aggregates[u] += weight_value
            node_aggregates[v] += weight_value

        nx.set_node_attributes(G, node_aggregates, name=weight)

        G = G.edge_subgraph(edges=G.top_k_edges(attribute=weight))
        return G


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    df = pd.read_csv('data.csv')
    graph = Graph.from_pandas_edgelist(df)
    graph = graph.subgraphX()
    graph.analysis()


# %%

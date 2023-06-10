from typing import Dict, List, Tuple
from typing import List
from collections import Counter
from typing import Optional, Dict
# Hint for Visual Code Python Interactive window
# %%

import logging
import types
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph
import numpy as np
import pandas as pd
from networkx.algorithms import approximation
from pandas.api.types import is_numeric_dtype
from collections import Counter, defaultdict


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


class EdgeView(nx.classes.reportviews.EdgeView):
    def sort(self,
             reverse: Optional[bool] = True,
             attribute: Optional[str] = 'weight'):
        sorted_edges = sorted(self(data=True),
                              key=lambda t: t[2].get(attribute, 1),
                              reverse=reverse)
        return {(u, v):_ for u, v, _ in sorted_edges}

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

    def layout(self, max_node_size: int = 300, max_edge_width: int = 10, max_font_size: int = 18):

        """
        Calculates the sizes for nodes, edges, and fonts based on node weights and edge weights.

        Parameters:
        - max_node_size (int): Maximum size for nodes (default: 300).
        - max_edge_width (int): Maximum width for edges (default: 10).
        - max_font_size (int): Maximum font size for node labels (default: 18).

        Returns:
        - Tuple[List[int], List[int], Dict[int, List[str]]]: A tuple containing the node sizes, edge widths,
          and font sizes for node labels.

        Raises:
        - ValueError: If the maximum values for node size, edge width, or font size are non-positive.
        """

        if max_node_size <= 0 or max_edge_width <= 0 or max_font_size <= 0:
            raise ValueError("Maximum values must be positive.")
        self.nodes_agg()
        # Normalize node weights between 0 and 1
        node_weights = [data.get('weight', 0)
                        for _, data in self.nodes(data=True)]
        min_node_weight = min(node_weights)
        max_node_weight = max(node_weights)
        if max_node_weight - min_node_weight == 0:
            normalized_weights = [0 for weight in node_weights]
        else:
            normalized_weights = [
                (weight - min_node_weight) / (max_node_weight - min_node_weight) for weight in node_weights]

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
        node_size_dict = dict(zip(self.nodes, normalized_weights))
        fonts_size = defaultdict(list)
        for node, width in node_size_dict.items():
            fonts_size[int(width * max_font_size) + 6].append(node)
        fonts_size = dict(fonts_size)

        return node_size_dict, edges_width, fonts_size

    def nodes_agg(self, attribute: Optional[str] = 'weight', operation: str = 'sum', normalized: bool = True) -> Dict[str, float]:
        """
        Calculates aggregate values for node attributes based on the edges in the graph.

        Parameters:
        - attribute (str): The attribute name to consider for aggregation (default: 'weight').
        - operation (str): The type of aggregation operation to perform. Valid operations: 'sum', 'min', 'max' (default: 'sum').
        - normalized (bool): Flag indicating whether to normalize the aggregate values between 0 and 1 (default: True).

        Returns:
        - Dict[str, float]: A dictionary containing the aggregate values for each node.

        Raises:
        - ValueError: If the specified attribute is not present in the graph edges.
        - ValueError: If an invalid operation is provided.
        """
        for node1, node2, data in self.edges(data=True):
            
            if attribute not in data:
                raise ValueError(f"Attribute '{attribute}' is not present in the graph edges.")
            else:
                break

        if operation == 'sum':
            aggregate_func = sum
        elif operation == 'min':
            aggregate_func = min
        elif operation == 'max':
            aggregate_func = max
        else:
            raise ValueError(f"Invalid operation: '{operation}'.")

        node_aggregates = {
            node: aggregate_func(data.get(attribute, 0) for _, _, data in self.edges(data=True) if node in (_, _)) for node in self.nodes}

        if normalized:
            min_val = min(node_aggregates.values())
            max_val = max(node_aggregates.values())
            node_aggregates = {
                node: (val - min_val) / (max_val - min_val) for node, val in node_aggregates.items()}

        nx.set_node_attributes(self, node_aggregates, attribute)

        return node_aggregates

    def subgraphX(self, node_list, max_edges):
        edges = list(nx.induced_subgraph(
            self, nbunch=node_list).edges.sort())[:max_edges]
        return nx.edge_subgraph(self, edges=edges)

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
        node_sizes, edge_widths, font_sizes = self.layout(300, 10, 18)
        pos = nx.spring_layout(self, seed=10396953)
        # nodes
        nx.draw_networkx_nodes(self,
                               pos,
                               ax=ax0,
                               node_size=list(node_sizes.values()),
                               #node_color=list(node_sizes.values()),
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
        node_list=self.nodes_circuits(node_list)
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

    @staticmethod
    def from_df( df_source: pd.DataFrame,
                 source_column: str = "source",
                 target_column: str = "target",
                 weight_column: str = "weight"):
        """
        Initialize netX instance with a simple dataframe

        :param df_source: DataFrame containing network data.
        :param source_column: Name of source nodes column in df_source.
        :param target_column: Name of target nodes column in df_source.
        :param weight_column: Name of edges weight column in df_source.

        """

        # Input validation
        if source_column not in df_source.columns:
            raise ValueError('missing source column')
        if target_column not in df_source.columns:
            raise ValueError('missing target column')
        if weight_column not in df_source.columns:
            raise ValueError('missing weight column')
        elif not is_numeric_dtype(df_source[weight_column]):
            raise ValueError('weight column is not numeric')

        # Preprocessing
        df = df_source.drop_duplicates()
        df.rename(columns={source_column: "source",
                  "target_column": "target", "weight_column": "weight"})
        df_node = pd.concat([df.groupby('source').agg(
            {'weight': 'sum'}), df.groupby('target').agg({'weight': 'sum'})])
        df_node = df_node.groupby(df_node.index).agg(
            {'weight': 'sum'}).sort_values(by="weight", ascending=False)
        df_node['name'] = df_node.index

        # Graph initialization
        G = Graph()
        logging.info(f'Adding {len(df_node)} nodes')
        G.add_nodes_from([(index, {"weight": row["weight"]})
                               for index, row in df_node.iterrows()])
        df = df[df['source'].isin(df_node['name']) &
                df['target'].isin(df_node['name'])]
        logging.info(f'Adding {len(df)} edges')
        edges = [(row["source"], row["target"],
                  {"weight": int(row["weight"])}) for index, row in df.iterrows()]
        G.add_edges_from(edges, weight="weight")

        # Graph metrics processing
        # Node
        nx.set_node_attributes(G, dict(G.degree), "degree")
        # Edge
        betweenness = nx.edge_betweenness_centrality(G, normalized=False)
        nx.set_edge_attributes(G, betweenness, "betweenness")
        return G


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    df = pd.read_csv('data.csv')
    graph = Graph.from_df(df)
    graph.analysis()


# %%

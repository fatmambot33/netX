 ##Hint for Visual Code Python Interactive window    
# %%

import logging
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from typing import Tuple,List
from pandas.api.types import is_numeric_dtype
from itertools import groupby
from operator import itemgetter
import numpy as np
class netX():
    """
    Class for representing and manipulating network data.
    """
    
    G:nx.Graph
    
    def __init__(self,df_source: pd.DataFrame,
                 source_column:str="source",
                 target_column:str="target",
                 weight_column:str="weight",
                 max_nodes:int=50):
        """
        Initialize netX instance with network data.

        :param df_source: DataFrame containing network data.
        :param source_column: Name of source nodes column in df_source.
        :param target_column: Name of target nodes column in df_source.
        :param weight_column: Name of edges weight column in df_source.
        :param max_nodes: Maximum number of nodes to include in the graph.
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
        df= df_source.drop_duplicates()
        df.rename(columns={source_column: "source", "target_column": "target","weight_column": "weight"})
        df['percentile'] = df["weight"].rank(pct = True,ascending=False)
        df_node=pd.concat([df.groupby('source').agg({'weight': 'sum'}),df.groupby('target').agg({'weight': 'sum'})])
        df_node=df_node.groupby(df_node.index).agg({'weight': 'sum'}).sort_values(by="weight",ascending=False).head(max_nodes)
        df_node['name']=df_node.index
        df_node['percentile'] = df_node["weight"].rank(pct = True,ascending=False)

        
        # Graph initialization
        self.G = nx.Graph()
        logging.info(f'Adding {len(df_node)} nodes')
        self.G.add_nodes_from([(index, {"size": row["weight"],"percentile":row["weight"]})
                for index, row in df_node.iterrows()])
        df=df[df['source'].isin(df_node['name']) & df['target'].isin(df_node['name'])]
        self.edge_width_mean = df.agg({'weight': 'mean'})
        logging.info(f'Adding {len(df)} edges')
        edges = [(row["source"], row["target"],
                  {"weight":int(row["weight"]),
                   "percentile":row["percentile"]}) for index, row in df.iterrows()]
        self.G.add_edges_from(edges,weight="weight")
        betweenness = nx.edge_betweenness_centrality(self.G, normalized=False)
        nx.set_edge_attributes(self.G, betweenness, "betweenness")

    def node_size(self,node_scale:int=10):
        """
        Calculate and return node sizes based on weight.
        """
        return [
        percentile * node_scale for percentile in nx.get_node_attributes(self.G, "percentile").values()]

    def edge_width(self,edge_scale:float=2.0):
        """
        Calculate and return node sizes based on weight.
        """
        return [
        percentile * edge_scale for percentile in nx.get_edge_attributes(self.G, "percentile").values()]

    def draw(self,scale:int=2,node_scale:int=10,edge_scale:float=2.0):
        """
        Draw the graph using matplotlib.
        """
        pos=nx.spring_layout(self.G,scale=scale)

        # nodes
        nodes_options = {"node_size": self.node_size(node_scale),
                        "node_color": self.node_size(1),
                        "cmap":plt.cm.Blues}
        nx.draw_networkx_nodes(self.G,
                            pos,
                            **nodes_options)

        # labels
        labels_options = {"alpha": 0.9,
                          }
        nx.draw_networkx_labels(
            self.G,
            pos,
            **labels_options)
        # edges
        edges_options = {"width":self.edge_width(edge_scale)}
        nx.draw_networkx_edges(
            self.G,
            pos,
            **edges_options)

        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()
    def analysis(self):
        import numpy as np


        degree_sequence = sorted((d for n, d in self.G.degree()), reverse=True)
        dmax = max(degree_sequence)

        fig = plt.figure("Degree of a random graph", figsize=(8, 8))
        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(5, 4)

        ax0 = fig.add_subplot(axgrid[0:3, :])
        Gcc = self.G.subgraph(sorted(nx.connected_components(self.G), key=len, reverse=True)[0])
        pos = nx.spring_layout(Gcc, seed=10396953)
        nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
        nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
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


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    import json
    with open('data.json') as f:
        df = pd.DataFrame(json.load(f))
    graph=netX(df)
    graph.analysis()
    graph.draw()
# %%
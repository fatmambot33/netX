import json
import logging
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pandas.api.types import is_numeric_dtype

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
        df = pd.concat([df, pd.DataFrame(
            {"source": df["target"], "target":df["source"], "weight":df["weight"]})], ignore_index=True).drop_duplicates()
        
        df_node = df.groupby('source').agg({'weight': 'sum'}).sort_values(by="weight").head(max_nodes)
        
        self.mean = df.agg({'weight': 'mean'})
        nodes = [(index, {"size": row["weight"]})
                for index, row in df_node.iterrows()]
        edges = [(row["source"], row["target"], int(row["weight"]))
                for index, row in df.iterrows() if row["source"] in nodes and row["target"] in nodes]
        
        # Graph initialization
        self.G = nx.Graph()
        logging.info(f'Adding {len(nodes)} nodes')
        self.G.add_nodes_from(nodes)
        logging.info(f'Adding {len(edges)} edges')
        self.G.add_weighted_edges_from(edges)

    def node_size(self):
        """
        Calculate and return node sizes based on weight.
        """
        return [
        v/self.mean * 10 for v in nx.get_node_attributes(self.G, "size").values()]

    def draw(self):
        """
        Draw the graph using matplotlib.
        """
        pos=nx.spring_layout(self.G,scale=2,k=5)

        # nodes
        nodes_options = {"node_size": self.node_size(),
                        "node_color": "tab:red",
                        "alpha": 0.9}
        nx.draw_networkx_nodes(self.G,
                            pos,
                            **nodes_options)

        # labels
        labels_options = {"alpha": 0.9}
        nx.draw_networkx_labels(
            self.G,
            pos,
            **labels_options)
        # edges
        edges_options = {
                        "alpha": 0.9,
                        "width": 8}
        nx.draw_networkx_edges(
            self.G,
            pos,
            **edges_options)

        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    # Opening JSON file
    with open('data.json') as f:
        df = pd.DataFrame(json.load(f))

    graph=netX(df)
    graph.draw()

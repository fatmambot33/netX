# %%
# Importing necessary libraries
import json
import logging

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pandas.api.types import is_numeric_dtype


# Defining the netX class
class netX():
    # This class will represent the Graph
    G:nx.Graph
    
    def __init__(self,df_source: pd.DataFrame,
                 source_column:str="source",
                 target_column:str="target",
                 weight_column:str="weight",
                 max_nodes:int=50):
        # Initialization function to create a netX object

        # Check if source_column exists in dataframe
        if source_column not in df_source.columns:
            raise ValueError('missing source column')
        
        # Check if target_column exists in dataframe
        if target_column not in df_source.columns:
            raise ValueError('missing target column')
        
        # Check if weight_column exists and is numeric
        if weight_column not in df_source.columns:
            raise ValueError('missing weight column')
        elif not is_numeric_dtype(df_source[weight_column]):
            raise ValueError('weight column is not numeric')

        # Drop duplicates and rename columns
        df= df_source.drop_duplicates()
        df.rename(columns={source_column: "source", target_column: "target",weight_column: "weight"})
        
        # Create symmetric dataframe and drop duplicates
        df = pd.concat([df, pd.DataFrame(
            {"source": df["target"], "target":df["source"], "weight":df["weight"]})], ignore_index=True).drop_duplicates()

        # Group by source, aggregate on weight and get the first max_nodes
        df_node = df.groupby('source').agg({'weight': 'sum'}).sort_values(by="weight").head(max_nodes)
        
        self.mean = df.agg({'weight': 'mean'})

        # Creating nodes
        nodes = [(index, {"size": row["weight"]})
                for index, row in df_node.iterrows()]
        # Creating edges
        edges = [(row["source"], row["target"], int(row["weight"]))
                for index, row in df.iterrows() if row["source"] in nodes and row["target"] in nodes]

        # Creating a Graph
        self.G = nx.Graph()
        
        logging.info(f'Adding {len(nodes)} nodes')
        self.G.add_nodes_from(nodes)
        
        logging.info(f'Adding {len(edges)} edges')
        self.G.add_weighted_edges_from(edges)

    def node_size(self):
        # Function to calculate node size based on weight
        return [
        v/self.mean * 10 for v in nx.get_node_attributes(self.G, "size").values()]

    def draw(self):
        # Function to draw the graph

        # Defining position of nodes and edges
        pos=nx.spring_layout(self.G,scale=2,k=5)

        # Drawing nodes with specific options
        nodes_options = {"node_size": self.node_size(),
                        "node_color": "tab:red",
                        "alpha": 0.9}
        nx.draw_networkx_nodes(self.G,
                            pos,
                            **nodes_options)

        # Drawing labels with specific options
        labels_options = {"alpha": 0.9}
        nx.draw_networkx_labels(
            self.G,
            pos,
            **labels_options)
        
        # Drawing edges with specific options
        edges_options = {
                        "alpha": 0.9,
                        "width": 8}
        nx.draw_networkx_edges(
            self.G,
            pos,
            **edges_options)

        # Setting margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()

# Main function to test the class functionality
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # Opening JSON file and loading it into a DataFrame
    with open('data.json') as f:
        df = pd.DataFrame(json.load(f))

    # Creating an object of netX class and drawing the graph
    graph=netX(df)
    graph.draw()
# %%

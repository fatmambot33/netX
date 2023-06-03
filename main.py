# %%
import json
import logging

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


class netX():
    G:nx.Graph
    def __init__(self,df_source: pd.DataFrame):
        df = df_source.drop_duplicates()
        df = pd.concat([df, pd.DataFrame(
            {"source": df["target"], "target":df["source"], "weight":df["weight"]})], ignore_index=True).drop_duplicates()
        df_node = df.groupby('source').agg({'value': 'sum'})
        self.mean = df.agg({'value': 'mean'})
        nodes = [(index, {"size": row["weight"]})
                for index, row in df_node.iterrows()]
        edges = [(row["source"], row["target"], int(row["weight"]))
                for index, row in df.iterrows()]
        self.G = nx.Graph()
        logging.info(f'Adding {len(nodes)} nodes')
        self.G.add_nodes_from(nodes)
        logging.info(f'Adding {len(edges)} edges')
        self.G.add_weighted_edges_from(edges)
    def node_size(self):
        return [
        v/self.mean * 10 for v in nx.get_node_attributes(self.G, "size").values()]

    def draw(self):
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
# %%

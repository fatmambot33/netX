netX

A Python package for analyzing and visualizing network data with NetworkX and Pandas.

Description

This Python package provides an interface for constructing, analyzing, and visualizing network data. It utilizes NetworkX for graph analysis, and Pandas for data manipulation. The library makes it easy to convert relational data into graph data and to visualize complex network relationships.

Requirements

Python 3.6+
NetworkX
Pandas
Matplotlib
json
logging
Usage

python
Copy code
# Import necessary modules
import json
import pandas as pd
from netX import netX

# Load your data
with open('data.json') as f:
    df = pd.DataFrame(json.load(f))

# Initialize the netX object
graph = netX(df)

# Draw the graph
graph.draw()
Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

License

MIT

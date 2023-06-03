# netX

## Description
`netX` is a Python package designed for the analysis and visualization of network data. This library leverages the powerful capabilities of NetworkX for graph analysis and Pandas for data manipulation, making it easier to convert relational data into graph data structures and to visually express complex network relationships.

## Requirements
Before getting started with `netX`, make sure you have the following Python packages installed:

- Python 3.6 or higher
- NetworkX
- Pandas
- Matplotlib
- json
- logging

You can usually install these with `pip`:

```bash
pip install networkx pandas matplotlib
```

## Usage

Here's a basic example of how to use netX:

```python
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
```
This will load data from data.json, create a graph structure from it, and then display it.

## Contributing

Contributions are welcome! If you have a feature request, bug report, or a new idea for netX, here's how you can contribute:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes in your new branch.
4. Submit a pull request with your changes.
5. Please make sure to update tests as appropriate.

## License

netX is licensed under the MIT license.

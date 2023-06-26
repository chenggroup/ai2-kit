# FAQ

## How to view atoms structures in Jupyter Notebook?

You can use the `ase.visualize.view` function to view atoms structures in Jupyter Notebook. If you would like to use `nglview` as your viewer, you need to take some extra steps to make it work.

```bash
# Install nglview
pip install nglview

# Enable Jupyter Notebook extension
jupyter nbextension enable --py widgetsnbextension
```
# Tips

## Use custom tags to simplify YAML configuration

`ai2-kit` implement some customized tags for YAML parser to simplify the configuration. 

### `!read`
`!read` can be used to read a file and use its content as the value of the tag. It accepts a list of strings as the argument. The strings will be joined together to form the path of the file to be read. For example,

```yaml
data: !read [/path/to/data/folder, data.csv]
```

### `!join`
`!join` can be used to join elements in a list into a single string. This is useful when you have to use a lot of absolution paths that shares the same based directory in your configuration. For example,

```yaml
.data_dir: &data_dir /data/in/a/very/long/path/
data1: !join [*data_dir, data1.csv]
data2: !join [*data_dir, data2.csv]
```
which is equivalent to

```yaml
data1: /data/in/a/very/long/path/data1.csv
data2: /data/in/a/very/long/path/data2.csv
```

## Use `nglview` to visualize atoms structures in Jupyter Notebook

You can use the `ase.visualize.view` function to view atoms structures in Jupyter Notebook. If you would like to use `nglview` as your viewer, you need to take some extra steps to make it work.

```bash
# Install nglview
pip install nglview

# Enable Jupyter Notebook extension
jupyter nbextension enable --py widgetsnbextension
```

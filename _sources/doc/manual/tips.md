# Tips
## Change logging level

You can change the logging level by setting the `LOG_LEVEL` environment variable. For example,

```bash
export LOG_LEVEL=DEBUG 
ai2-kit ...
```

## Path pattern

Most commands that take file or directory paths accept glob patterns, and they all share the same
pattern syntax, which is the standard Python glob syntax with one extension.

Besides the usual `*` and `?`, `**` matches any number of directories, so you don't need to know
how deep the files are nested:

```bash
# read every xyz file under workdir, no matter how deep it is
ai2-kit tool ase read './workdir/**/*.xyz' - write ./merged.xyz
```

> Note: always quote the pattern, or else your shell may expand it before `ai2-kit` sees it,
> and most shells don't support `**` in the way described here.

### The `/./` notation

Sometimes what you want to select is not the file that you can match, but a directory identified
by a file inside it. A typical case is a `deepmd/npy` dataset: the directory to read is the one
that contains a `type.raw` file, and there is no pattern for "directory containing X".

For this, `ai2-kit` supports a `/./` notation borrowed from `rsync`. The part after `/./` is a
literal path suffix that is joined to each match of the part before it, so you can match a file
and then navigate from it:

```bash
# match every type.raw under workdir, then go up to its parent directory,
# which is the dataset directory to read
ai2-kit tool dpdata read './workdir/**/type.raw/./..' --fmt deepmd/npy - write ./merged_dataset
```

The suffix is not a pattern, it is taken literally, and the joined path is normalized afterwards.
So `./workdir/**/set.000/./../..` selects the grandparent of every `set.000` directory. Paths that
do not exist after joining are dropped, and duplicated results are removed.

## Use custom tags to simplify YAML configuration

`ai2-kit` implement some customized tags for YAML parser to simplify the configuration. 

### `!load_text`
`!load_text` can be used to read a file and use its content as the value of the tag. It accepts a string value or a list of strings as the argument. The strings will be joined together to form the path of the file to be read. For example,

```yaml
data: !load_text /path/to/data/data.csv
# or
data: !load_text [/path/to/data, data.csv]
```

### `!load_yaml`
`!load_yaml` is same as `!load_text` except that it will parse content of the file as a YAML document. For example,

```yaml 
data: !load_yaml /path/to/data/data.yml
data: !load_yaml /path/to/data/data.json
# or
data: !load_yaml [/path/to/data, data.yml]
data: !load_yaml [/path/to/data, data.json]
```

> Note: it also support JSON format as it is a subset of YAML.

### `!join`
`!join` can be used to join elements in a list into a single string. This is useful when you have to use a lot of absolution paths that shares the same based directory in your configuration. For example,

> Note: this is string join not path join, so you need to add `/` manually when you want to join paths.

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

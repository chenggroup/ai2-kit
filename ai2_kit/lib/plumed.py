import pandas as pd
from typing import List, Union


def load_colvar_from_files(*files):
    """
    load plumed colvar from multiple files and concatenate into a single DataFrame
    """
    fragments = []
    for file in files:
        with open(file) as f:
            fragments.append(load_colvar(f))
    return pd.concat(fragments)


def load_colvar(fp, skip='#! FIELDS'):
    """
    load plumed colvar file as DataFrame

    :param fp: file pointer
    :param skip: skip the header line
    :return: DataFrame
    """

    line = next(fp)
    if not line.startswith(skip):
        raise ValueError(f'Invalid colvar file header: {line}')
    names = line[len(skip):].strip().split()
    return pd.read_csv(fp, delim_whitespace=True, names=names)


def get_cvs_bias_from_df(df, cv_names: Union[List[str], str], bias_name: str):
    """
    get collective variables and bias from pandas dataframe

    :param df: pandas dataframe
    :param cv_names: names of collective variable columns
    :param bias_name: name of bias column

    :return: cvs, bias in numpy array for gaussian_kde
    """
    if isinstance(cv_names, str):
        cv_names = [cv_names]

    if len(cv_names) == 1:
        cvs = df[cv_names[0]].values
    else:
        cvs = df[cv_names].values.T

    bias = df[bias_name].values
    return cvs, bias



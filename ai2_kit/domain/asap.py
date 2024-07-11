# Suppress numba warning of umap.
# Ref: https://github.com/lmcinnes/umap/issues/252
from numba.core.errors import NumbaDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

from asaplib.data.xyz import ASAPXYZ
from asaplib.cli.func_asap import cluster_process, set_reducer
from asaplib.hypers.hyper_soap import universal_soap_hyper
from asaplib.hypers.hyper_acsf import universal_acsf_hyper
from asaplib.reducedim.dim_reducer import Dimension_Reducers
from asaplib.cluster.ml_cluster_fit import LAIO_DB, sklearn_DB

from scipy.spatial.distance import cdist

import numpy as np



def get_descriptor(asapxyz: ASAPXYZ, setting: dict, keep_atomic=False, n_process=4, path_prefix='./asap-descriptor'):

    descriptor_spec = get_descriptor_spec(asapxyz, setting)
    asapxyz.compute_global_descriptors(
        desc_spec_dict = descriptor_spec,
        keep_atomic = keep_atomic,
        n_process = n_process,
    )
    asapxyz.save_state(path_prefix)

    global_descriptors = asapxyz.fetch_computed_descriptors(list(descriptor_spec.keys()))
    atomic_descriptors = asapxyz.fetch_computed_atomic_descriptors(list(descriptor_spec.keys())) if keep_atomic else None
    return global_descriptors, atomic_descriptors


def get_descriptor_spec(asapxyz: ASAPXYZ, setting: dict):
    descriptor_type, params = None, None
    for key in ['soap', 'acsf', 'cm']:
        params = setting.get(key, None)
        if params is not None:
            descriptor_type = key
            break
    else:
        raise ValueError('Unknown descriptor type')


    if descriptor_type == 'cm':
        return {'cm': {'type': 'CM'}}

    reducer_spec = dict(set_reducer(
        params['reducer_type'],
        params['element_wise'],
        params['zeta'],
    ))

    if descriptor_type == 'acsf':
        atomic_descriptor_spec = get_acsf_atomic_descriptor_spec(asapxyz, params)
    elif descriptor_type == 'soap':
        atomic_descriptor_spec = get_soap_atomic_descriptor_spec(asapxyz, params)

    descriptor_spec = {}
    for k, v in atomic_descriptor_spec.items():  # type: ignore
        descriptor_spec[k] = {
            'atomic_descriptor': {k: v},
            'reducer_function': reducer_spec,
        }
    return descriptor_spec


def get_soap_atomic_descriptor_spec(asapxyz: ASAPXYZ, params: dict):
    preset = params['preset']
    if preset is None:
        atomic_descriptor_spec = {
            'soap1': {
                'type': 'SOAP',
                'cutoff': params['r_cut'],
                'n': params['n_max'],
                'l': params['l_max'],
                'atom_gaussian_width': params['sigma'],
            }
        }
    else:
        atomic_descriptor_spec = universal_soap_hyper(asapxyz.get_global_species(), preset, dump=False)

    for k in atomic_descriptor_spec.keys():
        atomic_descriptor_spec[k]['rbf'] = params['rbf']
        atomic_descriptor_spec[k]['crossover'] = params['crossover']

    return atomic_descriptor_spec


def get_acsf_atomic_descriptor_spec(asapxyz: ASAPXYZ, params: dict):
    preset = params['preset']
    facsf_param = params['r_cut'] if preset is None else preset
    return universal_acsf_hyper(asapxyz.get_global_species(), facsf_param, dump=False)


def reduce_dimension(descriptors: np.ndarray, setting: dict):
    reducer = Dimension_Reducers(setting)
    reduced_descriptors = reducer.fit_transform(descriptors)
    return reduced_descriptors


def get_dbscan_trainer(descriptors: np.ndarray, metric='euclidean', eps=None, min_samples=2, eval_sample=50):
    if eps is None:
        n = len(descriptors)
        samples = descriptors[np.random.choice(n, min(n, eval_sample), replace=False)]
        # FIXME: the method to estimate eps is strange
        # FIXME: this will be broken when len of descriptors small
        eps = np.percentile(cdist(samples, descriptors, metric), min(100 * 10. / n, 99))  # type: ignore
    return sklearn_DB(eps, min_samples, metrictype=metric)


def get_laio_db_trainer(**kwargs):
    return LAIO_DB(**kwargs)


def get_trainer(descriptor: np.ndarray, setting: dict):
    dbscan_setting = setting.get('dbscan', None)
    if dbscan_setting is not None:
        return get_dbscan_trainer(descriptor, **dbscan_setting)
    laiodb_setting = setting.get('laiodb', None)
    if laiodb_setting is not None:
        return get_laio_db_trainer(**laiodb_setting)
    raise ValueError('Unknown trainer type')


def get_cluster(asapxyz: ASAPXYZ, descriptors: np.ndarray, trainer, path_prefix='./asap-cluster'):
    options = {
        'prefix': path_prefix,
        'savexyz': False,
        'savetxt': False,
    }
    labels: np.ndarray = cluster_process(asapxyz, trainer, descriptors, options)  # type: ignore

    groups = {}
    index = np.arange(len(labels))
    for label in set(labels):
        groups[label] = index[labels == label]
    return groups

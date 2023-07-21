from asaplib.data.xyz import ASAPXYZ
from asaplib.cli.func_asap import cluster_process, set_reducer
from asaplib.hypers.hyper_soap import universal_soap_hyper
from asaplib.hypers.hyper_acsf import universal_acsf_hyper
from asaplib.reducedim.dim_reducer import Dimension_Reducers
from asaplib.cluster.ml_cluster_fit import LAIO_DB, sklearn_DB

from scipy.spatial.distance import cdist

from ai2_kit.core.util import merge_dict
from ai2_kit.core.log import get_logger
import copy
import numpy as np

logger = get_logger(__name__)


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
    descriptor_type = setting.get('type', 'SOAP').upper()
    assert descriptor_type in ['SOAP', 'ACSF', 'CM'], f'Unknown descriptor type {descriptor_type}'

    if descriptor_type == 'CM':
        return {'cm': {'type': 'CM'}}

    reducer_spec = dict(set_reducer(
        setting['reducer_type'],
        setting['element_wise'],
        setting['zeta'],
    ))

    if descriptor_type == 'ACSF':
        atomic_descriptor_spec = get_acsf_atomic_descriptor_spec(asapxyz, setting)
    elif descriptor_type == 'SOAP':
        atomic_descriptor_spec = get_soap_atomic_descriptor_spec(asapxyz, setting)

    descriptor_spec = {}
    for k, v in atomic_descriptor_spec.items():  # type: ignore
        descriptor_spec[k] = {
            'atomic_descriptor': {k: v},
            'reducer_function': reducer_spec,
        }
    return descriptor_spec


def get_soap_atomic_descriptor_spec(asapxyz: ASAPXYZ, setting: dict):
    preset = setting['preset']
    if preset is None:
       atomic_descriptor_spec = {
            'soap1': {
                'type': 'SOAP',
                'cutoff': setting['r_cut'],
                'n': setting['n_max'],
                'l': setting['l_max'],
                'atom_gaussian_width': setting['sigma'],
            }
        }
    else:
        atomic_descriptor_spec = universal_soap_hyper(asapxyz.get_global_species(), preset, dump=False)

    for k in atomic_descriptor_spec.keys():
        atomic_descriptor_spec[k]['rbf'] = setting['rbf']
        atomic_descriptor_spec[k]['crossover'] = setting['crossover']

    return atomic_descriptor_spec


def get_acsf_atomic_descriptor_spec(asapxyz: ASAPXYZ, setting: dict):
    preset = setting['preset']
    facsf_param = setting['r_cut'] if preset is None else preset
    return universal_acsf_hyper(asapxyz.get_global_species(), facsf_param, dump=False)


def reduce_dimension(descriptors: np.ndarray, setting: dict):
    reducer = Dimension_Reducers(setting)
    reduced_descriptors = reducer.fit_transform(descriptors)
    return reduced_descriptors


def get_dbscan_trainer(descriptor: np.ndarray, metric='euclidean', eps=None, min_samples=2, eval_sample=50):
    if eps is None:
        n = len(descriptor)
        sample = descriptor[np.random.choice(n, min(n, eval_sample), replace=False)]
        # FIXME: the method to estimate eps is strange
        eps = np.percentile(cdist(sample, descriptor, metric), min(100 * 10. / n, 100))  # type: ignore
    return sklearn_DB(eps, min_samples, metrictype=metric)


def get_fdb_trainer():
    return LAIO_DB()


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


def test(path: str):
    from .constant import DEFAULT_ASAP_SOAP_DESC
    asapxyz = ASAPXYZ(path)
    setting = merge_dict(copy.deepcopy(DEFAULT_ASAP_SOAP_DESC), {'type': 'soap', 'preset': 'minimal'})
    global_descriptors, atomic_descriptor = get_descriptor(asapxyz, setting, keep_atomic=False)
    dim_reducer_setting = {
        'pca': {
            'type': 'PCA',
            'parameter': {
                'n_components': 3,
                'scalecenter': True,
            }
        }
    }
    dim_reducer_setting = {
        'tsne': {
            'type': 'TSNE',
            'parameter': {
                'perplexity': 30,
                'early_exaggeration': 12,
                'learning_rate': 200,
                'metric': 'euclidean',
            }
        }
    }
    reduced_descriptors = reduce_dimension(global_descriptors, dim_reducer_setting)
    trainer = get_dbscan_trainer(global_descriptors)
    r = get_cluster(asapxyz, reduced_descriptors, trainer)
    print(r)


if __name__ == '__main__':
    import fire
    fire.Fire(test)

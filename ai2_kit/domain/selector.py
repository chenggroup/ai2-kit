from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.log import get_logger
from ai2_kit.core.util import dump_json, dump_text, flush_stdio, limit
from ai2_kit.core.pydantic import BaseModel

from typing import List, Optional, Tuple, Dict
from io import StringIO
from dataclasses import dataclass
import pandas as pd
from tabulate import tabulate
from itertools import groupby
from functools import lru_cache
import traceback

import ase.io
import os

from .data import get_data_format, DataFormat, artifacts_to_ase_atoms
from .iface import ICllSelectorOutput, BaseCllContext
from .constant import DEFAULT_ASAP_SOAP_DESC, DEFAULT_ASAP_PCA_REDUCER


logger = get_logger(__name__)


class CllModelDeviSelectorInputConfig(BaseModel):

    class AsapOptions(BaseModel):
        disable: bool = False
        limit_per_cluster: int = 1
        """
        limit the number of structures to be selected from the same cluster
        """
        sort_by_ssw_energy: bool = False
        """
        sorted the structures by ssw_energy in each cluster
        """
        descriptor: dict = {'soap': { **DEFAULT_ASAP_SOAP_DESC, 'preset': 'minimal'}}
        dim_reducer: dict = {'pca': DEFAULT_ASAP_PCA_REDUCER}
        cluster: dict = {'dbscan': {}}

    f_trust_lo: float = 0.
    """
    the lower bound of model_devi score to select the structure for labeling
    """
    f_trust_hi: float = 65535.
    """
    the upper bound of model_devi score to select the structure for labeling
    """
    new_explore_system_q: float = 0.25
    """
    the quantile of model_devi score to select the structure for next round of exploration
    """
    asap_options: Optional[AsapOptions] = None
    """
    options for ASAP to further select candidates
    """
    screening_fn: Optional[str] = None
    """
    the function to screen the candidates, e.g
    "lambda x: x['ssw_energy'] < -1000"
    """
    max_decent_per_traj: int = -1
    """
    limit the max number of decent structures per trajectory, -1 means unlimited
    """
    workers: int = 4
    """
    number of workers to run the analysis
    """


@dataclass
class CllModelDevSelectorContext(BaseCllContext):
    ...


@dataclass
class CllModelDeviSelectorOutput(ICllSelectorOutput):
    candidates: List[Artifact]
    passing_rate: float
    new_explore_systems: List[Artifact]

    def get_model_devi_dataset(self):
        return self.candidates

    def get_passing_rate(self) -> float:
        return self.passing_rate

    def get_new_explore_systems(self) -> List[Artifact]:
        return self.new_explore_systems


@dataclass
class CllModelDeviSelectorInput:
    config: CllModelDeviSelectorInputConfig
    model_devi_data: List[Artifact]
    model_devi_file: str
    type_map: List[str]

    def set_model_devi_dataset(self, data: List[Artifact]):
        self.model_devi_data = data


async def cll_model_devi_selector(input: CllModelDeviSelectorInput, ctx: CllModelDevSelectorContext):
    executor = ctx.resource_manager.default_executor
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    executor.mkdir(work_dir)

    f_trust_lo = input.config.f_trust_lo
    f_trust_hi = input.config.f_trust_hi

    results = executor.run_python_fn(bulk_select_structures_by_model_devi)(
        model_devi_outputs=[a.to_dict() for a in input.model_devi_data],
        model_devi_file=input.model_devi_file,
        f_trust_lo=f_trust_lo, f_trust_hi=f_trust_hi,
        type_map=input.type_map, work_dir=work_dir,
        new_explore_system_q=input.config.new_explore_system_q,
        max_decent_per_traj=input.config.max_decent_per_traj,
        screening_fn=input.config.screening_fn,
        workers=input.config.workers,
    )

    candidates = [ result['decent'] for result, _ in results if 'decent' in result ]
    new_systems = [ result['next'] for result, _ in results if 'next' in result ]
    stats = [ stats for _, stats in results ]

    # group next_structures by `attrs.source` and keep the first one
    # so that the total number of explore structures will be the same as the original one
    get_source = lambda s: s['attrs']['source']
    # filter out the structures whose ancestor ends with '-fin'
    # FIXME: use a dedicated field to indicate the final structure
    new_systems = [s for s in new_systems if not s['attrs']['ancestor'].endswith('-fin')]
    new_systems = sorted(new_systems, key=get_source)
    new_systems = [next(group) for _source, group in groupby(
        new_systems, key=get_source)]

    # write model_deviation stats report
    # TODO: refactor into a function
    def _get_row(stat: dict):
        url = stat['src']
        url = f'...{url[-30:]}' if len(url) > 30 else url  # wrap url if it is too long
        total, good, decent, poor = stat['total'], stat['good'], stat['decent'], stat['poor']
        return [
            url, total, good, decent, poor,
            f'{good / total * 100:.2f}%', f'{decent / total * 100:.2f}%', f'{poor / total * 100:.2f}%',
        ]
    headers = ['file', 'total', 'good', 'decent', 'poor', 'good%', 'decent%', 'poor%']
    table = [ _get_row(stat) for stat in stats]
    total = sum(row[1] for row in table)
    total_good = sum(row[2] for row in table)

    stats_report = tabulate(table, headers=headers, tablefmt='tsv')
    logger.info('stats report: \n%s\n', stats_report)
    executor.dump_text(stats_report, os.path.join(work_dir, 'stats.tsv'))

    # further select candidates by ASAP
    if input.config.asap_options and not input.config.asap_options.disable:
        asap_options = input.config.asap_options
        candidates = executor.run_python_fn(bulk_select_distinct_structures)(
            candidates=candidates,
            descriptor_opt=asap_options.descriptor,
            dim_reducer_opt=asap_options.dim_reducer,
            cluster_opt=asap_options.cluster,
            type_map=input.type_map,
            work_dir=work_dir,
            limit_per_cluster=asap_options.limit_per_cluster,
            sort_by_energy=asap_options.sort_by_ssw_energy,
            workers=input.config.workers,
        )

    return CllModelDeviSelectorOutput(
        candidates=[Artifact.of(**a) for a in candidates],
        new_explore_systems=[Artifact.of(**a) for a in new_systems],
        passing_rate=total_good / total,
    )


def bulk_select_structures_by_model_devi(model_devi_outputs: List[ArtifactDict],
                                            model_devi_file: str,
                                            f_trust_lo: float,
                                            f_trust_hi: float,
                                            new_explore_system_q: float,
                                            type_map: List[str],
                                            work_dir: str,
                                            max_decent_per_traj: int,
                                            screening_fn: Optional[str],
                                            workers: int = 4,
                                            ) -> List[Tuple[Dict[str, ArtifactDict], dict]]:
    import joblib
    return joblib.Parallel(n_jobs=workers)(
        joblib.delayed(select_structures_by_model_devi)(
            model_devi_output=output,
            model_devi_file=model_devi_file,
            f_trust_lo=f_trust_lo, f_trust_hi=f_trust_hi,
            type_map=type_map,
            work_dir=os.path.join(work_dir, 'model_devi', f'{i:06}'),
            max_decent_per_traj=max_decent_per_traj,
            new_explore_system_q=new_explore_system_q,
            screening_fn=screening_fn,
        )
        for i, output in enumerate(model_devi_outputs)
    )  # type: ignore


def select_structures_by_model_devi(model_devi_output: ArtifactDict,
                                    model_devi_file: str,
                                    f_trust_lo: float,
                                    f_trust_hi: float,
                                    type_map: List[str],
                                    work_dir: str,
                                    new_explore_system_q: float,
                                    max_decent_per_traj: int,
                                    screening_fn: Optional[str],
                                    ) -> Tuple[Dict[str, ArtifactDict], dict]:
    """
    analysis the model_devi output of explore stage and select candidates

    :param next_explore_system_q: the quantile of model_devi score to select the structure for next round of exploration
    """
    os.makedirs(work_dir, exist_ok=True)
    dump_json(model_devi_output, os.path.join(work_dir, 'model_devi_output.debug.json'))

    model_devi_dir = model_devi_output['url']
    model_devi_file = model_devi_output['attrs'].pop('model_devi_file', model_devi_file)

    force_col = 'max_devi_f'
    print(f'criteria: {f_trust_lo} <= {force_col} < {f_trust_hi}')

    # get path of model_devi file
    data_format = get_data_format(model_devi_output)  # type: ignore
    if data_format in (DataFormat.LAMMPS_OUTPUT_DIR, DataFormat.LASP_LAMMPS_OUT_DIR):
        model_devi_file = os.path.join(model_devi_dir, model_devi_file)
    elif data_format == DataFormat.ANYWARE_OUTPUT_DIR:
        model_devi_file = os.path.join(model_devi_dir, 'model_devi.out')
    else:
        raise ValueError('unknown model_devi_data types')
    logger.info('start to analysis file: %s', model_devi_file)

    # load model_devi data
    with open(model_devi_file, 'r') as f:
        text = f.read()
    df = pd.read_csv(StringIO(text.lstrip('#')), delim_whitespace=True)
    # layout:
    #        step  max_devi_v  min_devi_v  avg_devi_v  max_devi_f  min_devi_f  avg_devi_f
    # 0        0    0.006793    0.000672    0.003490    0.143317    0.005612    0.026106
    # 1      100    0.006987    0.000550    0.003952    0.128178    0.006042    0.022608

    # load structures
    atoms_list = []
    if data_format == DataFormat.LAMMPS_OUTPUT_DIR:
        lammpstrj_file = model_devi_output['attrs'].pop('structures', 'traj.lammpstrj')
        atoms_list += ase.io.read(os.path.join(model_devi_dir, lammpstrj_file), ':', format='lammps-dump-text', specorder=type_map)
    elif data_format in (DataFormat.LASP_LAMMPS_OUT_DIR, DataFormat.ANYWARE_OUTPUT_DIR):
        structures_file = os.path.join(model_devi_dir, 'structures.xyz')
        atoms_list += ase.io.read(structures_file, ':', format='extxyz')
    else:
        raise ValueError('unknown model_devi_data types')

    # screening structure before model_devi analysis
    # FIXME: this should be moved to after the model_devi analysis
    if screening_fn is not None:
        if 'ssw_energy' in atoms_list[0].info:
            s_ssw_energy = pd.Series(map(lambda atoms: atoms.info['ssw_energy'], atoms_list))  # type: ignore

            # the following ssw_* methods are for the screening_fn
            # so don't need to worry about the unused warning
            ssw_energy_max = s_ssw_energy.max()
            ssw_energy_min = s_ssw_energy.min()
            def ssw_energy_quantile(q):
                # the quantile will be evaluated every time, which is not efficient
                # so here we cache the result, carefully
                return lru_cache()(s_ssw_energy.quantile)(q)
            ssw_enenrgy_quantile = ssw_energy_quantile

        # return the df row whose atoms pass the screening_fn
        _screening_fn = eval(screening_fn, locals())  # str to function
        df = df[[ _screening_fn(atoms) for atoms in atoms_list ]]


    # evaluate new found structures by their model_devi score in 3 levels: good, decent, poor
    good_df   = df[df[force_col] < f_trust_lo]
    decent_df = df[(df[force_col] >= f_trust_lo) & (df[force_col] < f_trust_hi)]
    poor_df   = df[df[force_col] >= f_trust_hi]

    # select the last frame from df whose model_devi score is less than the quantile
    # as the initial structure for next round of exploration to replace the original one
    # the next frame should be selected from good or decent frame
    # if there is no good or decent frame, use the first frame as the initial structure
    # NOTE: equal is essential to ensure the existence of next structure
    # NOTE: select the last frame can increase the diversity of structures
    _ndf = df[df[force_col] < f_trust_hi]
    if len(_ndf) == 0:
        next_df = df.head(1)  # the first frame is the initial structure
    else:
        next_df = _ndf[_ndf[force_col] <= _ndf[force_col].quantile(new_explore_system_q)].tail(1)

    stats = {
        'src': model_devi_file,
        'total': len(df),
        'good': len(good_df),
        'decent': len(decent_df),
        'poor': len(poor_df),
    }

    result: Dict[str, ArtifactDict] = {}
    # TODO: refactor the following repeating code
    # TODO: dump good may lead to storage issue, so disable it for now
    if len(good_df) > 0:
        good_file = os.path.join(work_dir, 'good.xyz')
        # ase.io.write(good_file, [atoms_list[_i] for _i in good_df.index], format='extxyz')
        result['good'] = {'url': good_file, 'format': DataFormat.EXTXYZ,  # type: ignore
                            'attrs': {**model_devi_output['attrs']}}

    if len(poor_df) > 0:
        poor_file = os.path.join(work_dir, 'poor.xyz')
        # ase.io.write(poor_file, [atoms_list[_i] for _i in poor_df.index], format='extxyz')
        result['poor'] = {'url': poor_file, 'format': DataFormat.EXTXYZ,  # type: ignore
                            'attrs': {**model_devi_output['attrs']}}
    if len(decent_df) > 0:
        decent_file = os.path.join(work_dir, 'decent.xyz')
        ase.io.write(decent_file,
                        list(limit((atoms_list[_i] for _i in decent_df.index), max_decent_per_traj)),
                        format='extxyz')
        result['decent'] = {'url': decent_file, 'format': DataFormat.EXTXYZ,  # type: ignore
                            'attrs': {**model_devi_output['attrs']}}
    if len(next_df) > 0:
        next_file = os.path.join(work_dir, 'next.xyz')
        ase.io.write(next_file, [atoms_list[_i] for _i in next_df.index], format='extxyz')
        result['next'] = {'url': next_file, 'format': DataFormat.EXTXYZ,  # type: ignore
                            'attrs': {**model_devi_output['attrs']}}
    dump_json([result, stats, list(decent_df.index), list(next_df.index)], os.path.join(work_dir, 'result.debug.json'))
    return result, stats


def bulk_select_distinct_structures(candidates: List[ArtifactDict],
                                    descriptor_opt: dict,
                                    dim_reducer_opt: dict,
                                    cluster_opt: dict,
                                    type_map: List[str],
                                    work_dir: str,
                                    limit_per_cluster: int = -1,
                                    sort_by_energy: bool = False,
                                    workers: int = 4,
                                    ) -> List[ArtifactDict]:
    try:
        dump_json(candidates, os.path.join(work_dir, 'candidates.debug.json'))
    except Exception as e:
        pass
    get_ancestor = lambda c: c['attrs']['ancestor']
    candidates = sorted(candidates, key=get_ancestor)
    inputs = []
    for i, (ancestor_key, candidate_group) in enumerate(groupby(candidates, key=get_ancestor)):
        candidate_group = list(candidate_group)
        inputs.append((candidate_group, candidate_group[0]['attrs']))

    import joblib
    return joblib.Parallel(n_jobs=workers)(
        joblib.delayed(select_distinct_structures)(
            candidates=group,
            attrs=attrs,
            descriptor_opt=descriptor_opt,
            dim_reducer_opt=dim_reducer_opt,
            cluster_opt=cluster_opt,
            type_map=type_map,
            work_dir=os.path.join(work_dir, 'asap', f'{i:06}'),
            limit_per_cluster=limit_per_cluster,
            sort_by_energy=sort_by_energy,
        ) for i, (group, attrs) in enumerate(inputs)
    )  # type: ignore


def select_distinct_structures(candidates: List[ArtifactDict],
                                attrs: dict,
                                descriptor_opt: dict,
                                dim_reducer_opt: dict,
                                cluster_opt: dict,
                                type_map: List[str],
                                work_dir: str,
                                limit_per_cluster: int = -1,
                                sort_by_energy: bool = False,
                                ):

    from .asap import get_descriptor, reduce_dimension, get_trainer, get_cluster
    from asaplib.data.xyz import ASAPXYZ

    os.makedirs(work_dir, exist_ok=True)

    dump_json(attrs, os.path.join(work_dir, 'attrs.debug.json'))
    # load structures and save it to a tmp file for ASAP to load
    atoms_list = [atoms for _, atoms in artifacts_to_ase_atoms(candidates, type_map=type_map)]


    if len(atoms_list) < 20:
        # FIXME: there are a lot of potential issue when the number of atoms is small
        # the root cause is in asaplib, which I guess has not been tested with small dataset
        selected_atoms_list = atoms_list
    elif limit_per_cluster <= 0:
        selected_atoms_list = atoms_list
    else:
        if sort_by_energy and 'ssw_energy' in atoms_list[0].info:
            atoms_list = sorted(atoms_list, key=lambda atoms: atoms.info['ssw_energy'])

        # use asaplib to group structures
        # load structures to ASAP
        tmp_structures_file = os.path.join(work_dir, '.tmp-structures.xyz')
        ase.io.write(tmp_structures_file, atoms_list, format='extxyz')
        asapxyz = ASAPXYZ(tmp_structures_file)

        # group structures
        try:
            asap_path_prefix = os.path.join(work_dir, 'asap')
            descriptors, _ = get_descriptor(asapxyz, descriptor_opt, path_prefix=asap_path_prefix)
            reduced_descriptors = reduce_dimension(descriptors, dim_reducer_opt)
            trainer = get_trainer(reduced_descriptors, cluster_opt)
            cluster_labels = get_cluster(asapxyz, reduced_descriptors, trainer, path_prefix=asap_path_prefix)

            # dump_json(cluster_labels, os.path.join(work_dir, 'cluster.debug.json'))
            selected_frames = []
            for frames in cluster_labels.values():
                if len(frames) < limit_per_cluster:
                    selected_frames += list(frames)
                else:
                    selected_frames += list(frames[:limit_per_cluster])
            selected_atoms_list = [atoms_list[i] for i in selected_frames]
        except Exception as e:
            print('asaplib failed: %s', e)
            # dump exception to file
            dump_text(traceback.format_exc(), os.path.join(work_dir, 'asaplib-exception.txt'))
            selected_atoms_list = atoms_list

    # write selected structures to file
    distinct_structures_file = os.path.join(work_dir,  'distinct_structures.xyz')
    ase.io.write(distinct_structures_file, selected_atoms_list, format='extxyz')

    output = {
        'url': distinct_structures_file,
        'format': DataFormat.EXTXYZ,
        'attrs': attrs,
    }
    dump_json(output, os.path.join(work_dir, 'output.debug.json'))
    flush_stdio()  # flush joblib stdio buffer
    return output

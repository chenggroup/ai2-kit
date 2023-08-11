from asaplib.data.xyz import ASAPXYZ
from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.log import get_logger
from ai2_kit.core.util import flat_evenly, dump_json, flush_stdio

from typing import List, Optional, Tuple
from io import StringIO
from pydantic import BaseModel
from dataclasses import dataclass
import pandas as pd
from tabulate import tabulate
from itertools import groupby
from operator import itemgetter

import ase.io
import os
import numpy as np

from .data import get_data_format, DataFormat, artifacts_to_ase_atoms
from .iface import ICllSelectorOutput, BaseCllContext
from .constant import LAMMPS_DUMP_DIR, LAMMPS_DUMP_SUFFIX, DEFAULT_ASAP_SOAP_DESC, DEFAULT_ASAP_PCA_REDUCER
from .asap import get_descriptor, reduce_dimension, get_trainer, get_cluster


logger = get_logger(__name__)


class CllModelDeviSelectorInputConfig(BaseModel):
    class AsapOptions(BaseModel):
        disable: bool = False
        max_structures_per_system: int = 10
        """
        max number of structures to be selected from the same group
        """
        descriptor: dict = {'soap': { **DEFAULT_ASAP_SOAP_DESC, 'preset': 'minimal'}}
        dim_reducer: dict = {'pca': DEFAULT_ASAP_PCA_REDUCER}
        cluster: dict = {'dbscan': {}}

    f_trust_lo: float
    f_trust_hi: float
    asap_options: Optional[AsapOptions]


@dataclass
class CllModelDevSelectorContext(BaseCllContext):
    ...


@dataclass
class CllModelDeviSelectorOutput(ICllSelectorOutput):
    candidates: List[Artifact]
    passing_rate: float

    def get_model_devi_dataset(self):
        return self.candidates

    def get_passing_rate(self) -> float:
        return self.passing_rate


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
    )

    candidates = [candidate for candidate, _ in results if candidate is not None]  # filter out None
    stats = [stats for _, stats in results]

    # write model_deviation stats report
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
            max_structures_per_system=asap_options.max_structures_per_system,
        )

    return CllModelDeviSelectorOutput(
        candidates=[ Artifact.of(**a) for a in candidates ],
        passing_rate=total_good / total,
    )


def __export_remote_functions():

    def bulk_select_structures_by_model_devi(model_devi_outputs: List[ArtifactDict],
                                             model_devi_file: str,
                                             f_trust_lo: float,
                                             f_trust_hi: float,
                                             type_map: List[str],
                                             work_dir: str,
                                             workers: int = 4,
                                             ) -> List[Tuple[Optional[ArtifactDict], dict]]:
        import joblib
        return joblib.Parallel(n_jobs=workers)(
            joblib.delayed(select_structures_by_model_devi)(
                model_devi_output=output,
                model_devi_file=model_devi_file,
                f_trust_lo=f_trust_lo, f_trust_hi=f_trust_hi,
                type_map=type_map,
                work_dir=os.path.join(work_dir, 'model_devi', f'{i:06}')
            )
            for i, output in enumerate(model_devi_outputs)
        )  # type: ignore


    def select_structures_by_model_devi(model_devi_output: ArtifactDict,
                                        model_devi_file: str,
                                        f_trust_lo: float,
                                        f_trust_hi: float,
                                        type_map: List[str],
                                        work_dir: str,
                                        ) -> Tuple[Optional[ArtifactDict], dict]:
        """
        analysis the model_devi output of explore stage and select candidates
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

        # evaluate new found structures by their model_devi score in 3 levels: good, decent, poor
        good_df   = df[df[force_col] < f_trust_lo]
        decent_df = df[(df[force_col] >= f_trust_lo) & (df[force_col] < f_trust_hi)]
        poor_df   = df[df[force_col] >= f_trust_hi]

        stats = {
            'src': model_devi_file,
            'total': len(df),
            'good': len(good_df),
            'decent': len(decent_df),
            'poor': len(poor_df),
        }

        # write decent structures into ase atoms and write to file
        decent_structures_artifact = None
        if len(decent_df) > 0:
            model_devi_decent_file = os.path.join(work_dir, 'model_devi_decent.xyz')
            if data_format == DataFormat.LAMMPS_OUTPUT_DIR:
                lammps_dump_dir = model_devi_output['attrs'].pop('lammps_dump_dir', LAMMPS_DUMP_DIR)
                atoms_list = []
                for frame_id in decent_df.step:
                    dump_file = os.path.join(model_devi_dir, lammps_dump_dir, f'{frame_id}{LAMMPS_DUMP_SUFFIX}')
                    atoms_list.extend(ase.io.read(dump_file, ':', format='lammps-dump-text', specorder=type_map))
            elif data_format == DataFormat.LASP_LAMMPS_OUT_DIR:
                structures_file = os.path.join(model_devi_dir, 'structures.xyz')
                atoms_list = list(itemgetter(*decent_df.step)(ase.io.read(structures_file, ':', format='extxyz')))  # type: ignore
            else:
                raise ValueError('unknown model_devi_data types')
            ase.io.write(model_devi_decent_file, atoms_list, format='extxyz')
            decent_structures_artifact = {
                'url': model_devi_decent_file,
                'format': DataFormat.EXTXYZ,
                'attrs': {
                    **model_devi_output['attrs'],
                }
            }
        dump_json([decent_structures_artifact, stats], os.path.join(work_dir, 'output.debug.json'))
        return decent_structures_artifact, stats  # type: ignore

    def bulk_select_distinct_structures(candidates: List[ArtifactDict],
                                        descriptor_opt: dict,
                                        dim_reducer_opt: dict,
                                        cluster_opt: dict,
                                        type_map: List[str],
                                        work_dir: str,
                                        max_structures_per_system: int = -1,
                                        workers: int = 4,
                                        ) -> List[ArtifactDict]:

        inputs = []
        for i, (ancestor_key, candidate_group) in enumerate(groupby(candidates, key=lambda c: c['attrs']['ancestor'])):
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
                max_structures_per_system=max_structures_per_system,
            ) for i, (group, attrs) in enumerate(inputs)
        )  # type: ignore


    def select_distinct_structures(candidates: List[ArtifactDict],
                                   attrs: dict,
                                   descriptor_opt: dict,
                                   dim_reducer_opt: dict,
                                   cluster_opt: dict,
                                   type_map: List[str],
                                   work_dir: str,
                                   max_structures_per_system: int = -1,
                                   ):
        os.makedirs(work_dir, exist_ok=True)

        dump_json(attrs, os.path.join(work_dir, 'attrs.debug.json'))
        # load structures and save it to a tmp file for ASAP to load
        atoms_list = [atoms for _, atoms in artifacts_to_ase_atoms(candidates, type_map=type_map)]

        if len(atoms_list) < max(max_structures_per_system, 10):
            # Nothing to do when the total number of atoms is small
            # FIXME: there are a lot of potential issue when the number of atoms is small
            # the root cause is in asaplib, which I guess is not tested with small dataset
            selected_atoms_list = atoms_list
        else:
            tmp_structures_file = os.path.join(work_dir, '.tmp-structures.xyz')
            ase.io.write(tmp_structures_file, atoms_list, format='extxyz')

            # use asaplib to group structures
            asapxyz = ASAPXYZ(tmp_structures_file)

            # group structures
            path_prefix = os.path.join(work_dir, 'asap')
            descriptors, _ = get_descriptor(asapxyz, descriptor_opt, path_prefix=path_prefix)
            reduced_descriptors = reduce_dimension(descriptors, dim_reducer_opt)
            trainer = get_trainer(reduced_descriptors, cluster_opt)
            cluster_labels = get_cluster(asapxyz, reduced_descriptors, trainer, path_prefix=path_prefix)

            # if max_structures_per_system is not set
            # selected at least 1 structure from each group
            if max_structures_per_system <= 0:
                max_structures_per_system = len(cluster_labels)

            # unpack clusters into a list and
            # ensure frames of the same group won't be next to each other
            # so that we can pick up distinct structures by simply choose the first N frames
            order_frames = flat_evenly(cluster_labels.values())
            selected_frames = order_frames[:max_structures_per_system]
            selected_atoms_list = [atoms_list[i] for i in selected_frames]

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

    return (
        bulk_select_structures_by_model_devi,
        bulk_select_distinct_structures,
    )

(
    bulk_select_structures_by_model_devi,
    bulk_select_distinct_structures,
) = __export_remote_functions()

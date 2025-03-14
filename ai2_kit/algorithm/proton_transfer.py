from MDAnalysis import Universe
from MDAnalysis.lib.distances import minimize_vectors
from multiprocessing.pool import Pool
import ase.io as ai
from ase import Atom
from functools import partial

from dataclasses import dataclass
from typing import List, Tuple, NamedTuple, Union

import fire
import numpy as np
from numpy import mean
import matplotlib.pyplot as plt

import json
import os
import io

# TODO: use array instead of list for better performance when possible
# TODO: use numba to speed up the calculation


class AnalysisResult(NamedTuple):
    indicator_position: Tuple[float, float, float]
    transfers: List[Tuple[int, int]]


@dataclass
class SystemInfo:  # Information about the system
    initial_donor: int
    u: Universe
    cell: List[float]
    acceptor_elements: List[str]


# The relevant parameters of the algorithm
class AlgorithmParameter(NamedTuple):
    r_a: float  # The radius used to search for acceptor
    r_h: float  # The radius used to search for H
    rho_0: float  # Control the rate of the weights change
    rho_max: float  # The critical value of proton transfer
    max_depth: int  # The maximum length of the path
    g_threshold: float  # The threshold for whether to join the node to the path


class System(object):
    def __init__(self, sys_info: SystemInfo, parameter: AlgorithmParameter):
        self.u = sys_info.u
        self.cell = sys_info.cell
        self.r_a = parameter.r_a
        self.r_h = parameter.r_h
        self.g_threshold = parameter.g_threshold
        self.max_depth = parameter.max_depth
        self.rho_0 = parameter.rho_0
        self.rho_max = parameter.rho_max
        self.acceptor_elements = sys_info.acceptor_elements

    def frame_analysis(self, prev_donor: int, acceptor_query: str, time: int):
        self.u.trajectory[time]
        donor = prev_donor
        transfers = []
        list_of_paths = [[prev_donor]]
        list_of_weights = [[1]]
        for depth in range(self.max_depth):
            for j, path in enumerate(list_of_paths):
                found = False
                if depth == len(path) - 1:
                    acceptors = self.u.select_atoms(
                        f"(around {self.r_a} index {path[-1]}) and ({acceptor_query})")
                    protons = self.u.select_atoms(
                        f"(around {self.r_h} index {path[-1]}) and (name H)")
                    for i, acceptor in enumerate(acceptors.ix):
                        g, proton = self.calculate_g(
                            path[-1], acceptor, protons.ix)
                        if (g >= self.g_threshold) and (acceptor not in path):
                            found = True
                            list_of_weights.append(
                                list_of_weights[j] + [g * list_of_weights[j][-1]])
                            list_of_paths.append(path + [acceptor])
                            if proton > 0 and all(w >= 0.9 for w in list_of_weights[j]):
                                donor = acceptor
                                transfers.append((int(acceptor), int(proton)))
                if found:
                    list_of_paths.pop(j)
                    list_of_weights.pop(j)
        indicator_position = self.calculate_position(
            list_of_paths, list_of_weights)
        result = AnalysisResult(indicator_position=tuple(
            indicator_position[0]), transfers=transfers)
        return donor, result

    def calculate_g(self, donor: int, acceptor: int, protons: list):
        donor_pos = self.u.atoms[donor].position
        acceptor_pos = self.u.atoms[acceptor].position
        g_value = 0
        proton_index = -1
        for i, proton in enumerate(protons):
            proton_pos = self.u.atoms[proton].position
            r_da = minimize_vectors(acceptor_pos - donor_pos, self.cell)
            r_dh = minimize_vectors(proton_pos - donor_pos, self.cell)
            z1 = np.dot(r_dh, r_da)
            z2 = np.dot(r_da, r_da)
            z = (z1 / z2)
            p = ((self.rho_max - z) / (self.rho_max - self.rho_0))
            if p >= 1:
                g = 0
            elif p <= 0:
                g = 1
                proton_index = protons[i]
            else:
                g = -6 * (p ** 5) + 15 * (p ** 4) - 10 * (p ** 3) + 1
            g_value = g_value + g
        return g_value, proton_index

    def calculate_position(self, paths: list, weights: list):
        positions_all = []
        nodes_all = []
        weights_all = []
        for i, path in enumerate(paths):
            for j, node in enumerate(path):
                if node not in nodes_all:
                    donor_pos = self.u.atoms[path[0]].position
                    if j == 0:
                        positions_all.append(donor_pos)
                    else:
                        acceptor_pos = self.u.atoms[node].position
                        min_vector = minimize_vectors(
                            acceptor_pos - donor_pos, self.cell)
                        real_acceptor_pos = min_vector + donor_pos
                        positions_all.append(real_acceptor_pos)
                    nodes_all.append(node)
                    weights_all.append(weights[i][j])
                else:
                    index = nodes_all.index(node)
                    weights_all[index] = max(weights[i][j], weights_all[index])
        p = np.array(positions_all).reshape(-1, 3)
        w = np.array(weights_all).reshape(1, -1)
        z = w @ p
        pos_ind = z / w.sum()
        return pos_ind

    def analysis(self, initial_donor: int, out_dir: str):
        donor = initial_donor
        acceptor_query = ' or '.join(
            [f'(name {el})' for el in self.acceptor_elements])
        rand_file = io.FileIO(os.path.join(
            out_dir, f'{initial_donor}.jsonl'), 'w')
        writer = io.BufferedWriter(rand_file)
        line = (tuple(self.u.atoms[initial_donor].position.astype(float)), [])
        writer.write((json.dumps(line) + '\n').encode('utf-8'))
        for time in range(self.u.trajectory.n_frames-1):
            donor, result = self.frame_analysis(donor, acceptor_query, time+1)
            line = (result.indicator_position, result.transfers)
            writer.write((json.dumps(line) + '\n').encode('utf-8'))
        writer.flush()


def proton_transfer_detection_cli(
    input_traj: str,
    out_dir: str,
    cell: List[float],
    acceptor_elements: List[str],
    initial_donors: List[int],
    core_num: int = 4,
    dt: float = 0.0005,
    r_a: float = 4.0,
    r_h: float = 1.3,
    rho_0: float = 1 / 2.2,
    rho_max: float = 0.5,
    max_depth: int = 4,
    g_threshold: float = 0.0001,
):
    """
    cli entry for proton transfer detection
    """
    # setup universe from trajectory
    universe = Universe(input_traj)
    universe.trajectory.ts.dt = dt
    universe.dimensions = np.array(cell)

    proton_transfer_detection(
        universe,
        out_dir,
        acceptor_elements,
        initial_donors,
        core_num,
        r_a,
        r_h,
        rho_0,
        rho_max,
        max_depth,
        g_threshold,
    )


def proton_transfer_detection(
    universe: Universe,
    out_dir: str,
    acceptor_elements: List[str],
    initial_donors: Union[List[int], np.ndarray],
    core_num: int = 4,
    r_a: float = 4.0,
    r_h: float = 1.3,
    rho_0: float = 1 / 2.2,
    rho_max: float = 0.5,
    max_depth: int = 4,
    g_threshold: float = 0.0001,
):
    assert universe.dimensions is not None, "Universe dimensions is None"

    os.makedirs(out_dir, exist_ok=True)

    sys_info = SystemInfo(
        initial_donor=-1,
        u=universe,
        cell=universe.dimensions,
        acceptor_elements=acceptor_elements,
    )

    parameter = AlgorithmParameter(
        r_a=r_a,
        r_h=r_h,
        rho_0=rho_0,
        rho_max=rho_max,
        max_depth=max_depth,
        g_threshold=g_threshold,
    )

    system = System(
        sys_info,
        parameter,
    )

    with Pool(processes=core_num) as pool:
        pool.map(partial(system.analysis, out_dir=out_dir), initial_donors)


def visualize_transfer(analysis_result: str, input_traj: str, output_traj: str, initial_donor: int, cell: list):
    stc_list = ai.read(input_traj, index=":")
    donor = initial_donor
    with open(os.path.join(analysis_result, f'{initial_donor}.jsonl'), mode='r') as reader:
        for i, line in enumerate(reader):
            line = json.loads(line)
            stc_list[i][donor].symbol = 'N'
            stc_list[i].set_cell(cell)
            stc_list[i].set_pbc(True)
            if line[1]:
                donor = line[1][-1][0]
            pos = line[0]
            ind = Atom('F', pos)
            stc_list[i].append(ind)
    ai.write(output_traj, stc_list)


def analysis_transfer_paths(analysis_result: str, initial_donor: int):
    donor = initial_donor
    print("transfer_paths")
    fmt = "{:^40}\t{:^8}"
    content = fmt.format("transfer_path_index", "Snapshot")
    print(f"{content}")
    with open(os.path.join(analysis_result, f'{initial_donor}.jsonl'), mode='r') as reader, \
            open(os.path.join(analysis_result, f'{initial_donor}_proton_infos.jsonl'), mode='wb') as writer:
        for i, line in enumerate(reader):
            line = json.loads(line)
            if line[1]:
                for j, event in enumerate(line[1]):
                    acceptor = event[0]
                    proton = event[1]
                    content = f"{donor}({proton})->"
                    donor = acceptor
                writer.write((json.dumps((proton, i)) + '\n').encode('utf-8'))
                content = content + f"{acceptor}"
                content = fmt.format(f"{content}", f"{i}")
                print(content)
        if proton:
            writer.write((json.dumps((proton, i+1)) + '\n').encode('utf-8'))


def calculate_distances(analysis_result: str, input_traj: str, upper_index: List[int], lower_index: List[int], initial_donor: int, interval: int = 1):
    stc_list = ai.read(input_traj, index=":")
    upper_pos = [stc_list[0][i].position[2] for i in upper_index]
    lower_pos = [stc_list[0][i].position[2] for i in lower_index]
    upper_interface = mean(upper_pos)
    lower_interface = mean(lower_pos)
    start = 0
    with open(os.path.join(analysis_result, f'{initial_donor}_proton_infos.jsonl'), mode='rb') as reader, \
            open(os.path.join(analysis_result, f'{initial_donor}_proton_distance_to_interface.jsonl'), mode='wb') as writer:
        for i, line in enumerate(reader):
            proton_info = json.loads(line)
            end = proton_info[1]
            for j in range(start, end, interval):
                distance = min(abs(stc_list[j][proton_info[0]].position[2] - upper_interface),
                               abs(stc_list[j][proton_info[0]].position[2] - lower_interface))
                writer.write((json.dumps(distance) + '\n').encode('utf-8'))
            start = end


def show_distance_change(analysis_result: str, initial_donor: int):
    y = []
    with open(os.path.join(analysis_result, f'{initial_donor}_proton_distance_to_interface.jsonl'), mode='rb') as reader:
        for i, line in enumerate(reader):
            y.append(json.loads(line))
    if y:
        x = np.arange(len(y))
        plt.plot(x, y)
        plt.xlabel("Time")
        plt.ylabel("Distance")
        plt.title("Proton distance to interface")
        plt.savefig(os.path.join(analysis_result, f'{initial_donor}_proton_distance_to_interface.png'))


def detect_type_change(analysis_result: str, atom_types: dict, donors: list):
    type_change_event = []
    type_list = []
    type_change_name = []
    for i in range(len(atom_types)):
        for j in range(i + 1):
            type_change_event.append(([], []))
            type_list.append([j, i])
            type_change_name.append(
                f"{list(atom_types.keys())[j]}<->{list(atom_types.keys())[i]}")
    for donor in donors:
        with open(os.path.join(analysis_result, f'{donor}.jsonl'), mode='r') as reader:
            change_times = 0
            change_info = []
            change = False
            for k in range(len(atom_types)):
                if donor in list(atom_types.values())[k]:
                    change_info.append((donor, k, 0))
                    change_times = 1
            for i, line in enumerate(reader):
                line = json.loads(line)
                if line[1]:
                    for j, event in enumerate(line[1]):
                        for k in range(len(atom_types)):
                            if event[0] in list(atom_types.values())[k]:
                                change_info.append((event[0], k, i))
                                change_times = change_times + 1
                                change = True
                        if change == False and change_times > 0:
                            change_info.append((event[0], -1, i))
                        if change_times == 2:
                            real_type = change_info[-1][1]
                            type = [min(change_info[0][1], change_info[-1][1]),
                                    max(change_info[0][1], change_info[-1][1])]
                            index = [x[0] for x in change_info]
                            time = [x[2] for x in change_info]
                            if index not in type_change_event[type_list.index(type)][0]:
                                type_change_event[type_list.index(
                                    type)][0].append(index)
                                type_change_event[type_list.index(
                                    type)][1].append(time)
                            change_info = [(index[-1], real_type, time[-1])]
                            change_times = 1
                            change = False
    for j, type_change in enumerate(type_change_event):
        type_change[0].sort(key=lambda x: len(x))
        type_change[1].sort(key=lambda x: len(x))
    print("proton transfer type change")
    print("-------------------------------------")
    fmt = "{:^25}\t{:^15}\t{:^15}"
    content = fmt.format("Path_index", "start_Snapshot", "end_Snapshot")
    print(f"{content}")
    for i in range(len(type_change_name)):
        print(type_change_name[i])
        for j in range(len(type_change_event[i][0])):
            content = ' -> '.join([f'{el}' for el in type_change_event[i][0][j]])
            content = fmt.format(f"{content}", f"{type_change_event[i][1][j][0]}",
                                 f"{type_change_event[i][1][j][-1]}")
            print(f"{content}")


cli_entry = {
    'analyze': proton_transfer_detection_cli,
    'visualize': visualize_transfer,
    'show-transfer-paths': analysis_transfer_paths,
    'show-type-change': detect_type_change,
    'calculate-distances': calculate_distances,
    'show-distance-change': show_distance_change,
}


if __name__ == '__main__':
    fire.Fire(cli_entry)

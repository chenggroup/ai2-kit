from MDAnalysis.lib.distances import distance_array
import MDAnalysis as mda
import numpy as np
import fire
import os


def _ecn_analysis_per_frame(frame_index, reference_group, configuration_group, universe):
    l_avg, ecn = 0, 0
    universe.trajectory[frame_index]
    distance_mat = distance_array(reference=reference_group, configuration=configuration_group, box=universe.dimensions)
    # Calculate l_av (average bond length).
    n_in, n_o = distance_mat.shape
    l_min_array = np.min(distance_mat, axis=1)
    l_av = np.divide(distance_mat,l_min_array[:, np.newaxis])
    l_av = np.exp(1.0 - np.power(l_av, 6))
    denominator = np.sum(l_av, axis=1)
    l_av = np.divide(np.multiply(distance_mat, l_av), denominator[:, np.newaxis])
    l_av = np.sum(l_av, axis=1)
    l_av = np.mean(l_av)
    # Calculate ECN (effective coordination number)
    ecn_part = l_av.copy()
    ecn_part = np.exp(1.0- np.power(np.divide(distance_mat, ecn_part), 6))
    ecn_part = np.sum(ecn_part)/n_in
    l_avg += l_av
    ecn += ecn_part
    return frame_index, l_avg, ecn


def ecn_analysis(input_traj: str, out_dir: str, center: str, ligand: str, cell: list[float]):
    """
    Calculate ECN (effective coordination number) and l_av (average bond length).

    :param input_traj: path of input trajectory file
    :param out_dir: path of output directory
    :param center: central atoms, in the format of MDA atom selection string, e.g. 'name In'
    :param ligand: atoms that surround the central atoms, in the format of MDA atom selection string, e.g. 'name O'
    :param cell: cell dimensions
    """

    universe = mda.Universe(input_traj)
    universe.dimensions = np.array(cell)
    atomgroup = universe.atoms
    reference_group = atomgroup.select_atoms(center)
    configuration_group = atomgroup.select_atoms(ligand)
    # TODO: this can be parallelized
    results = []
    for k in range(universe.trajectory.n_frames):
        result = _ecn_analysis_per_frame(frame_index=k,
                                         reference_group=reference_group,
                                         configuration_group=configuration_group,
                                         universe=universe)
        results.append(result)
    results_arr = np.array(results)
    avr_arr = np.mean(results_arr, axis=0)

    # dump results
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, 'raw.txt'), results_arr,
               fmt='    %d        %.4f   %.4f',
               delimiter='   ', header='frame_index   l_av   ECN')
    with open(os.path.join(out_dir, 'stats.txt'), mode='w') as fp:
        fp.write('\n'.join([f'l_av = {avr_arr[1]:.4f}',
           f'ECN = {avr_arr[2]:.4f}']))


def _count_shared_polyhedra_per_frame(frame_index, reference_group, configuration_group, universe, cutoff, coordination_num = -1):
    universe.trajectory[frame_index]

    distance_mat = distance_array(reference=reference_group, configuration=configuration_group, box=universe.dimensions)
    neighbor_mat = distance_mat < cutoff
    neighbor_mat = neighbor_mat.astype(int)
    # common_neighbor_mat: a matrix of coordination and shared oxygen numbers.
    # The former numbers are diagonal and the later are other elements.
    common_neighbor_mat = np.dot(neighbor_mat, neighbor_mat.T)

    # Counting polyhedra with particular coordination number.
    if coordination_num > 0:
        for i in range(reference_group.n_atoms):
            if common_neighbor_mat[i][i] != coordination_num:
                common_neighbor_mat[i] = 0
                common_neighbor_mat[:, i] = 0
    # counting
    common_neighbor_mat = common_neighbor_mat[np.triu_indices_from(common_neighbor_mat, k=1)]
    corner = np.count_nonzero(common_neighbor_mat == 1)
    edge = np.count_nonzero(common_neighbor_mat == 2)
    face = np.count_nonzero(common_neighbor_mat>=3)
    return frame_index, corner, edge, face


def count_shared_polyhedra(input_traj: str, out_dir: str, center: str, ligand: str, cell: list[float], cutoff: float, coord_num: int = -1):
    """
    Count the number of shared polyhedra in each frame of the trajectory.

    :param input_traj: path of input trajectory file
    :param out_dir: path of output directory
    :param center: central atoms, in the format of MDA atom selection string, e.g. 'name In'
    :param ligand: atoms that surround the central atoms, in the format of MDA atom selection string, e.g. 'name O'
    :param cell: cell dimensions
    :param cutoff: cutoff distance
    :param coord_num: coordination number
    """
    universe = mda.Universe(input_traj)
    universe.dimensions = np.array(cell)
    atomgroup = universe.atoms
    reference_group = atomgroup.select_atoms(center)
    configuration_group = atomgroup.select_atoms(ligand)

    # TODO: this can be parallelized
    results = []
    for k in range(universe.trajectory.n_frames):
        result = _count_shared_polyhedra_per_frame(frame_index=k,
                                                   reference_group=reference_group,
                                                   configuration_group=configuration_group,
                                                   universe=universe,
                                                   cutoff=cutoff, coordination_num=coord_num)
        results.append(result)

    results_arr = np.array(results)
    avg_arr = np.mean(results_arr, axis=0)

    # dump results
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, 'raw.txt'), results_arr,
               fmt='    %d        %.4f   %.4f   %.4f',
               delimiter='   ', header='frame_index   corner   edge   face')

    with open(os.path.join(out_dir, 'stats.txt'), mode='w') as fp:
        fp.write('\n'.join([f'Corner-share = {avg_arr[1]:.4f}',
                            f'Edge-share = {avg_arr[2]:.4f}',
                            f'Face-share = {avg_arr[3]:.4f}', ]))


cli_entry = {
    'ecn-analysis': ecn_analysis,
    'count-shared-polyhedra': count_shared_polyhedra,
}


if __name__ == '__main__':
    fire.Fire(cli_entry)

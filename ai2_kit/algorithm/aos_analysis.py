import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np
import os
import fire


# l_av and ECN
def analysis_ecn_per_frame(frame_index, reference_group, configuration_group, universe):
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
        return [frame_index, l_avg, ecn]

def effective_cn_analysis(input_traj: str, result_dir: str, ref: str, conf: str, cell: list[float]):
    # Generate data for writing
    data_array = []
    os.makedirs(result_dir, exist_ok=True)
    universe = mda.Universe(input_traj)
    universe.dimensions = np.array(cell)
    atomgroup = universe.atoms
    reference_group = atomgroup.select_atoms(ref)
    configuration_group = atomgroup.select_atoms(conf)
    # for loop: to analyze a trajectory
    for k in range(universe.trajectory.n_frames):
        data_array.append(analysis_ecn_per_frame(frame_index=k, reference_group=reference_group, configuration_group=configuration_group, universe=universe))
    data_array = np.array(data_array)
    avg_array = np.mean(data_array, axis=0)
    # Write data
    np.savetxt(f'{result_dir}/result', data_array, fmt='    %d        %.4f   %.4f', delimiter='   ', header='frame_index   l_av   ECN')
    with open(f'{result_dir}/average', mode='w') as writer:
        writer.write('\n'.join([f'l_av = {avg_array[1]:.4f}', 
           f'ECN = {avg_array[2]:.4f}']))
        writer.close()



# count polyhedral
def analysis_per_frame(frame_index, reference_group, configuration_group, universe, cutoff, coordination_num = -1):
        universe.trajectory[frame_index]
        distance_mat = distance_array(reference=reference_group, configuration=configuration_group, box=universe.dimensions)
        neighbor_mat = distance_mat < cutoff
        # change [True/False] into [1/0]
        neighbor_mat = neighbor_mat + 0
        # common_neighbor_mat: a matrix of coordination and shared oxygen numbers. The former numbers are diagonal and the later are other elements.
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
        return [frame_index, corner, edge, face]


def count_shared_polyhedra(input_traj: str, result_dir: str, ref: str, conf: str, cell: list[float], cutoff: float, coord_num: int):
    universe = mda.Universe(input_traj)
    universe.dimensions = np.array(cell)
    atomgroup = universe.atoms
    reference_group = atomgroup.select_atoms(ref)
    configuration_group = atomgroup.select_atoms(conf)
    
    # Generate data for writing
    data_array = []
    os.makedirs(result_dir, exist_ok=True)
    # for loop: to analyze a trajectory
    for k in range(universe.trajectory.n_frames):
        data_array.append(analysis_per_frame(frame_index=k, reference_group=reference_group, configuration_group=configuration_group, universe=universe, cutoff=cutoff, coordination_num=coord_num))
    data_array = np.array(data_array)
    avg_array = np.mean(data_array, axis=0)
    # Write data
    np.savetxt(f'{result_dir}/result', data_array, fmt='    %d        %.4f   %.4f   %.4f', delimiter='   ', header='frame_index   corner   edge   face')
    with open(f'{result_dir}/average', mode='w') as writer:
        writer.write('\n'.join([f'Corner-share = {avg_array[1]:.4f}', 
           f'Edge-share = {avg_array[2]:.4f}',
           f'Face-share = {avg_array[3]:.4f}',
           ]))
        writer.close()
        




if __name__ == '__main__':
    fire.Fire({
        'ECN': effective_cn_analysis,
        'count-shared-polyhedra': count_shared_polyhedra,
    })






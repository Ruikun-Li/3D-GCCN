# 3D-GCNN: 3D Vascular Connectivity Graph Construction
# Ruikun Li

import os
import time
import numpy as np
import SimpleITK as sitk
import networkx as nx
import skfmm
import pickle as pkl
from utils import check_and_create_path
import sys
sys.path.append('./')

img_dir = '../Image_folder/'
prob_dir = '../Label_folder/'
graph_save_dir = '3d_graph/'
check_and_create_path(graph_save_dir)
num_class = 1

window_size = 12  # sampling interval
edge_dist_thresh = 16  # travel time threshold
distance_type = 'geo'  # 'geo' or 'eu'


def load_data(data_path):
    data = sitk.ReadImage(data_path)
    data = sitk.GetArrayFromImage(data)
    return data


if __name__ == '__main__':
    time_start = time.time()
    test_list = os.listdir(img_dir)
    print('Processing on {} images.'.format(len(test_list)))

    for index, test_file in enumerate(test_list):
        time01 = time.time()
        print('  Generating graph of', test_file, '...')
        img_path = img_dir + test_file + '/patient.nii.gz'
        prob_path = prob_dir + test_file + '/hepaticvessel.nii.gz'

        img = load_data(img_path).astype(np.float32)
        vessel_prob = load_data(prob_path).astype(np.float32)

        # padding
        z_size = img.shape[0]
        padding_z = (16 - z_size % 16) % 16
        img = np.pad(img, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        vessel_prob = np.pad(vessel_prob, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        y_size = img.shape[1]
        padding_y = (16 - y_size % 16) % 16
        img = np.pad(img, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        vessel_prob = np.pad(vessel_prob, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        x_size = img.shape[2]
        padding_x = (16 - x_size % 16) % 16
        img = np.pad(img, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        vessel_prob = np.pad(vessel_prob, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')

        print(img.shape, vessel_prob.shape)

        # Find max
        img_z = img.shape[0]
        img_y = img.shape[1]
        img_x = img.shape[2]
        z_quan = range(0, img_z, int(window_size / 2))
        z_quan = sorted(list(set(z_quan) | set([img_z])))
        # z_quan = sorted(list(set(z_quan)))
        y_quan = range(0, img_y, window_size)
        y_quan = sorted(list(set(y_quan) | set([img_y])))
        # y_quan = sorted(list(set(y_quan)))
        x_quan = range(0, img_x, window_size)
        x_quan = sorted(list(set(x_quan) | set([img_x])))
        # x_quan = sorted(list(set(x_quan)))

        max_val = []
        max_pos = []
        num_node = 0
        for z_idx in range(len(z_quan) - 1):
            for y_idx in range(len(y_quan) - 1):
                for x_idx in range(len(x_quan) - 1):
                    cur_patch = vessel_prob[z_quan[z_idx]:z_quan[z_idx+1], y_quan[y_idx]:y_quan[y_idx+1], x_quan[x_idx]:x_quan[x_idx+1]]
                    # Choose center voxel if current patch is 0
                    if np.sum(cur_patch) == 0:
                        max_val.append(0)
                        max_pos.append((z_quan[z_idx]+int(cur_patch.shape[0]/2), y_quan[y_idx]+int(cur_patch.shape[1]/2), x_quan[x_idx]+int(cur_patch.shape[2]/2)))
                    else:
                        max_val.append(np.max(cur_patch))
                        temp = np.zeros(3)
                        count = 0
                        for i_0 in range(cur_patch.shape[0]):
                            for i_1 in range(cur_patch.shape[1]):
                                for i_2 in range(cur_patch.shape[2]):
                                    if cur_patch[i_0, i_1, i_2] == 1:
                                        temp += np.array([i_0, i_1, i_2])
                                        count += 1
                        temp = np.around(temp / count).astype(np.int)
                        max_pos.append((z_quan[z_idx] + temp[0], y_quan[y_idx] + temp[1], x_quan[x_idx] + temp[2]))
                    num_node += 1

        # Generate a graph
        graph = nx.Graph()
        for node_idx, (node_z, node_y, node_x) in enumerate(max_pos):
            graph.add_node(node_idx, kind='MP', z=node_z, y=node_y, x=node_x, label=node_idx)

        speed = vessel_prob
        node_list = list(graph.nodes)
        num_edge = 0
        for i, n in enumerate(node_list):
            if speed[graph.nodes[n]['z'], graph.nodes[n]['y'], graph.nodes[n]['x']] == 0:
                continue
            # If the values in 3*3 neighbor are very small
            neighbor = speed[max(0, graph.nodes[n]['z'] - 1):min(img_z, graph.nodes[n]['z'] + 2),
                       max(0, graph.nodes[n]['y'] - 1):min(img_y, graph.nodes[n]['y'] + 2),
                       max(0, graph.nodes[n]['x'] - 1):min(img_x, graph.nodes[n]['x'] + 2)]
            if np.mean(neighbor) < 0.5:
                continue

            if distance_type == 'geo':
                # Define the geodesic distances using travel time
                phi = np.ones_like(speed)
                phi[graph.nodes[n]['z'], graph.nodes[n]['y'], graph.nodes[n]['x']] = -1.0
                # print(np.max(speed), np.min(speed), speed[graph.node[n]['z'], graph.node[n]['y'], graph.node[n]['x']])
                # tt = skfmm.travel_time(phi=phi, speed=speed, narrow=0)  # travel time
                try:
                    # this commonly fails in skfmm due to numerics issues
                    tt = skfmm.travel_time(phi=phi, speed=speed, order=2)
                except:
                    # fall back to order=1
                    tt = skfmm.travel_time(phi=phi, speed=speed, order=1)
                # Caculate the geodesic distances for subsequent nodes
                for n_comp in node_list[i+1:]:
                    geo_dist = tt[graph.nodes[n_comp]['z'], graph.nodes[n_comp]['y'], graph.nodes[n_comp]['x']]  # travel time
                    if geo_dist < edge_dist_thresh:
                        graph.add_edge(n, n_comp, weight=edge_dist_thresh/(edge_dist_thresh + geo_dist))
                        num_edge += 1

            elif distance_type == 'eu':
                # Define the euclidean distances threshold
                edge_dist_thresh_sq = edge_dist_thresh ** 2
                # Caculate the euclidean distances for subsequent nodes
                for n_comp in node_list[i + 1:]:
                    eu_dist = (graph.nodes[n_comp]['z']-graph.nodes[n]['z'])**2 + (graph.nodes[n_comp]['y']-graph.nodes[n]['y'])**2 + (graph.nodes[n_comp]['x']-graph.nodes[n]['x'])**2
                    if eu_dist < edge_dist_thresh_sq:
                        graph.add_edge(n, n_comp, weight=1)
                        num_edge += 1

            else:
                raise NotImplementedError

        print('Generate total', num_node, 'nodes, ', num_edge, 'edges.')

        # Save the graph as files
        graph_save_path = os.path.join(graph_save_dir, test_file + '_' + str(window_size) + '.gpickle')
        nx.write_gpickle(graph, graph_save_path, protocol=pkl.HIGHEST_PROTOCOL)
        graph.clear()

        print('********** Time:', time.time() - time01, '**********')

    print('\n Total time:', time.time() - time_start)
    print('Average time:', (time.time() - time_start) / len(test_list))


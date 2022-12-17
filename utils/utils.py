# Utils for 3D-GCNN

import os
import numpy as np
import sys
from torch.utils import data
import random
import SimpleITK as sitk
from skimage import transform
import networkx as nx


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def load_data(data_path):
    data = sitk.ReadImage(data_path)
    data = sitk.GetArrayFromImage(data)
    data = np.clip(data, 0, 400)
    # data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = (data - np.mean(data)) / np.std(data)
    return data


def check_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


class MyDataset(data.Dataset):
    def __init__(self, instance_list, data_dir, label_dir, graph_dir, valid_flag=False):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.graph_dir = graph_dir
        self.instance_list = instance_list
        self.valid_flag = valid_flag

    def __getitem__(self, index):
        instance_name = self.instance_list[index]
        data_path = self.data_dir + instance_name + '/patient.nii.gz'
        label_path = self.label_dir + instance_name + '/hepaticvessel.nii.gz'
        graph_path = self.graph_dir + instance_name + '_12.gpickle'

        # load the data and label
        data = load_data(data_path)
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label)

        # padding
        z_size = data.shape[0]
        padding_z = (16 - z_size % 16) % 16
        data = np.pad(data, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        label = np.pad(label, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        y_size = data.shape[1]
        padding_y = (16 - y_size % 16) % 16
        data = np.pad(data, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        label = np.pad(label, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        x_size = data.shape[2]
        padding_x = (16 - x_size % 16) % 16
        data = np.pad(data, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        label = np.pad(label, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        # print(data.shape, label.shape)

        # for training
        if not self.valid_flag:
            # More data augmentation can be used according to the specific task
            if random.randint(0, 1):
                rotate_degree = random.uniform(-5, 5)
                data = transform.rotate(data.transpose(1, 2, 0), angle=rotate_degree, resize=False, cval=-1)
                data = data.transpose(2, 0, 1)
                label = transform.rotate(label.transpose(1, 2, 0), angle=rotate_degree, resize=False, cval=0)
                label = label.transpose(2, 0, 1)
            if random.randint(0, 1):
                shift_value = random.uniform(-0.1, 0.1)
                scale_value = random.uniform(0.9, 1.1)
                data = data * scale_value + shift_value

            # load graph, get adjacency_matrix, graph data and label
            graph = nx.read_gpickle(graph_path)
            adj = nx.adjacency_matrix(graph).astype(np.float32).todense()  # (N, N)
            adj = np.array(adj).astype(np.float16)
            node_list = list(graph.nodes)
            graph_label = np.zeros([len(node_list), 1])
            patch_nodes = []
            for i, nd in enumerate(node_list):
                graph_label[i, 0] = label[graph.nodes[nd]['z'], graph.nodes[nd]['y'], graph.nodes[nd]['x']]
                patch_nodes.append([graph.nodes[nd]['z'], graph.nodes[nd]['y'], graph.nodes[nd]['x']])
            graph_label = np.array(graph_label).astype(np.float32)

            data = data[np.newaxis, :, :, :].astype(np.float32)
            label = (label > 0)[np.newaxis, :, :, :].astype(np.float32)
            graph_label = graph_label.astype(np.float32)
            patch_nodes = np.array(patch_nodes).astype(np.int)
            graph.clear()

            return data, label, patch_nodes, adj, graph_label

        # for validation
        else:
            data = data[np.newaxis, :, :, :].astype(np.float32)
            label = (label > 0)[np.newaxis, :, :, :].astype(np.float32)

            return data, label

    def __len__(self):
        return len(self.instance_list)


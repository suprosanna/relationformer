"""Functionality for 2D road network dataset."""

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from PIL import Image

import time
import pickle
import random
import yaml
import json


class ToulouseRoadNetworkDataset(Dataset):
    """
    Generates a subclass of the PyTorch torch.utils.data.Dataset class
    """
    def __init__(
        self, root_path="data/", split="valid", use_raw_images=False
    ):
        """
        :param root_path: root data path
        :param split: data split in {"train", "valid", "test", "augment"}
        :param max_prev_node: only return the last previous 'max_prev_node' elements in the adjacency row of a node
            default is 4, which corresponds to the 95th percentile in the data
        :param step: step size used in the data generation, default is 0.001° (around 110 metres per datapoint)
        :param use_raw_images: loads raw images if yes, otherwise faster and more compact numpy array representations
        :param return_coordinates: returns coordinates on the real map for each datapoint, used for qualitative studies
        """
        assert split in {"train", "valid", "test", "augment"}
        print(f"Started loading the data ({split})...")
        start_time = time.time()
        
        dataset_path = f"{root_path}/{split}.pickle"
        images_path = f"{root_path}/{split}_images.pickle"
        images_raw_path = f"{root_path}/{split}/images/"
        
        ids, list_nodes, list_edges = load_dataset(dataset_path)
        
        self.ids = ["{:0>7d}".format(int(i)) for i in ids]
        self.nodes = list_nodes
        self.edges = list_edges

        
        print(f"Started loading the images...")
        
        if use_raw_images:
            self.images = load_raw_images(ids, images_raw_path)
        else:
            self.images = load_images(ids, images_path)
        
        print(f"Dataset loading completed, took {round(time.time() - start_time, 2)} seconds!")
        print(f"Dataset size: {len(self)}\n")
    
    def __len__(self):
        r"""
        :return: data length
        """
        return len(self.ids)
    
    def __getitem__(self, idx):
        r"""
        :param idx: index in the data
        :return: chosen data point
        """
        return self.images[idx][None], self.nodes[idx], self.edges[idx], self.ids[idx]


def load_dataset(dataset_path):
    """
    Loads the chosen split of the data
    
    :param dataset_path: path of the data split pickle
    :param max_prev_node: only return the last previous 'max_prev_node' elements in the adjacency row of a node
    :param return_coordinates: returns coordinates on the real map for each datapoint
    :return:
    """
    with open(dataset_path, "rb") as pickled_file:
        dataset = pickle.load(pickled_file)
    
    list_nodes = []
    list_edges = []
    ids = list(dataset.keys())
    random.Random(42).shuffle(ids)  # permute to remove any correlation between consecutive datapoints
    
    for id in ids:
        datapoint = dataset[id]

        # Retrieve from dataset
        nodes = (torch.FloatTensor(datapoint['nodes']) + 1) / 2
        edges = torch.tensor((datapoint['edges']))[:, :2]

        # Sort edges
        edges_sorted = torch.sort(edges, 1)[0]
        edges_sorted = torch.stack(sorted(edges_sorted, key=lambda a: (a[0], a[1])))

        # Get rid of duplicate edges and nodes that do not participate in edges
        edges_clean = torch.unique(edges_sorted, dim=0)
        participating_nodes = torch.unique(edges_clean)

        for node in torch.arange(nodes.shape[0]-1, -1, -1):
            if node not in participating_nodes:
                edges_clean[edges_clean > node] -= 1

        nodes_clean = nodes[participating_nodes]

        list_nodes.append(nodes_clean)
        list_edges.append(edges_clean)
        
    return ids, list_nodes, list_edges

def load_images(ids, images_path):
    """
    Load images from arrays in pickle files
    
    :param ids: ids of the images in the data order
    :param images_path: path of the pickle file
    :return: the images, as pytorch tensors
    """
    images = []
    with open(images_path, "rb") as pickled_file:
        images_features = pickle.load(pickled_file)
    for id in ids:
        img = torch.FloatTensor(images_features["{:0>7d}".format(int(id))])
        assert img.shape[1] == img.shape[2]
        assert img.shape[1] in {64}
        images.append(img)
    
    return images

def load_raw_images(ids, images_path):
    """
    Load images from raw files
    
    :param ids: ids of the images in the data order
    :param images_path: path of the raw images
    :return: the images, as pytorch tensors
    """
    images = []
    for count, id in enumerate(ids):
        # if count % 10000 == 0:
        #     print(count)
        image_path = images_path + "{:0>7d}".format(int(id)) + ".png"
        img = Image.open(image_path).convert('L')
        img = tvf.to_tensor(img)
        assert img.shape[1] == img.shape[2]
        assert img.shape[1] in {64, 128}
        images.append(img)
    return images


def build_road_network_data(config, mode='split'):
    if mode == 'split':
        train_ds = ToulouseRoadNetworkDataset(
            root_path=config.DATA.DATA_PATH, split='train'
        )
        val_ds = ToulouseRoadNetworkDataset(
            root_path=config.DATA.DATA_PATH, split='valid'
        )

        return train_ds, val_ds
    elif mode == 'test':
        test_ds = ToulouseRoadNetworkDataset(
            root_path=config.DATA.DATA_PATH, split='test'
        )
        return test_ds


if __name__ == '__main__':
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence

    def custom_collate_fn(batch):
        """
        Custom collate function ordering the element in a batch by descending length
        
        :param batch: batch from pytorch dataloader
        :return: the ordered batch
        """
        x_adj, x_coord, y_adj, y_coord, img, seq_len, ids = zip(*batch)
        
        x_adj = pad_sequence(x_adj, batch_first=True, padding_value=0)
        x_coord = pad_sequence(x_coord, batch_first=True, padding_value=0)
        y_adj = pad_sequence(y_adj, batch_first=True, padding_value=0)
        y_coord = pad_sequence(y_coord, batch_first=True, padding_value=0)
        img, seq_len = torch.stack(img), torch.stack(seq_len)
        
        seq_len, perm_index = seq_len.sort(0, descending=True)
        x_adj = x_adj[perm_index]
        x_coord = x_coord[perm_index]
        y_adj = y_adj[perm_index]
        y_coord = y_coord[perm_index]
        img = img[perm_index]
        ids = [ids[perm_index[i]] for i in range(perm_index.shape[0])]
        
        return x_adj, x_coord, y_adj, y_coord, img, seq_len, ids

    class obj:
        def __init__(self, dict1):
            self.__dict__.update(dict1)

    def dict2obj(dict1):
        return json.loads(json.dumps(dict1), object_hook=obj)
    
    config = "configs/road_2D_deform_detr.yaml"
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)

    train_ds, val_ds = build_road_network_data(config, mode='split')
    # dataloader = DataLoader(train_ds, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    for i in [14, 2, 4, 6, 30, 26, 43, 24, 69173, 48360, 60201]:
        ret = train_ds[i]   # some strange cases 14, 2, 4, 6, 30, 26, 43, 24
        print(ret[-1])

        nodes_pixels = (ret[1] * ret[0].shape[-1]).type(torch.int32).numpy()

        image = ret[0].squeeze().cpu().numpy()
        image = np.flip(image, 0).copy()
        
        for node in nodes_pixels:
            image = cv2.circle(image, node, 5, (0, 0, 0), 2)
            cv2.imshow('testing', image)
            cv2.waitKey()

        print(ret[-2])
       
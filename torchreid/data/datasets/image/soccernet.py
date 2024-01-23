from __future__ import division, print_function, absolute_import
import glob
import os.path as osp
import os
import numpy as np
import time
from configparser import ConfigParser


side_map_train = ['4', '6', '9']
side_map_test = ['7', '8', '11']

from ..dataset import ImageDataset

class SoccerNet(ImageDataset):
    """Synergy Dataset.
    """
    _junk_pids = [0, -1]
    dataset_dir = 'reid_dataset/reid_dataset'
    dataset_url = None
    masks_base_dir = 'masks'

    masks_dirs = {
        # dir_name: (masks_stack_size, contains_background_mask)
        'reid_dataset/reid_dataset/masks': (36, False, '.npy')
    }
    
    def gallery_filter(self, q_pid, q_camid, q_ann, g_pids, g_camids, g_anns):
        """ Remove gallery samples that have the same pid and camid as the query sample, since ReID is a cross-camera
        person retrieval task for most datasets. However, we still keep samples from the same camera but of different
        identity as distractors."""
        remove = (g_camids != q_camid)
        return remove

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in SoccerNet.masks_dirs:
            return None
        else:
            return SoccerNet.masks_dirs[masks_dir]

    def infer_masks_path(self, img_path): # FIXME remove when all datasets migrated
        masks_path = img_path[:-4] + self.masks_suffix
        return masks_path

    def infer_keypoints_path(self, img_path): # FIXME remove when all datasets migrated
        masks_path = img_path + self.keypoints_suffix
        return masks_path

    def infer_segmentation_path(self, img_path): # FIXME remove when all datasets migrated
        masks_path = img_path + self.segmentation_suffix
        return masks_path

    def __init__(self, root='', masks_dir=None, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        #self.download_dataset(self.dataset_dir, self.dataset_url)
        self.masks_dir = masks_dir
 
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None
        
       
        np.random.seed(0)
        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        self.train_dir = osp.join(self.data_dir, 'train2')
        self.query_dir = osp.join(self.data_dir, 'query2')
       # self.gallery_dir = osp.join(self.data_dir, 'gallery2')
        self.gallery_dir = 'dataset/'

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]

        self.check_before_run(required_files)

        train, _ = self.process_dir(self.train_dir, 1)
        gallery, mapping = self.process_dir(self.gallery_dir, 2)
        query, _ = self.process_dir(self.query_dir, 3, mapping)


        super(SoccerNet, self).__init__(train, train, gallery, **kwargs)

    def process_dir(self, main_path, mode, mapping=[]):

        sequences_dir_list = [f for f in os.scandir(main_path) if f.is_dir()][48:]
        #if mode == 2: print(sequences_dir_list)
        data = []
        data2 = []
        id_dict = {}
        for seq_dir in sequences_dir_list:
            seq_name = seq_dir.name
            dir_path = osp.join(main_path, seq_name)
           # if mode != 2: continue

            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
            for img_path in img_paths:
                index = img_path.split('/')[-1][:-4].split('_')[0]
                pid = img_path.split('/')[-1][:-4].split('_')[1]
                camid = img_path.split('/')[-1][:-4].split('_')[2]
                game_id = img_path.split('/')[-1][:-4].split('_')[-2]
                role, side = img_path.split('/')[-1][:-4].split('_')[-1].split('-')

               # if role != 'player': continue
               # if game_id != '7': continue
               # if  camid != '116': continue

                team_id = 0
                if side == 'right': team_id = 1
                if mode == 1: team_id += side_map_train.index(game_id) * 2
                else: team_id += side_map_test.index(game_id) * 2

                #if role != 'player': continue
                #if side == 'right':
                   # if mode == 1: pid = side_map_train.index(camid) * 2
                   # else: pid = side_map_test.index(camid) * 2
                #elif side == 'left':
                   # if mode == 1: pid = side_map_train.index(camid) * 2 + 1
                   # else: pid = side_map_test.index(camid) * 2 + 1
                #else: continue

                masks_path = self.infer_masks_path(osp.join(self.masks_dir, 'SNMOT-' + camid, img_path.split('/')[-1]))
                
                if (int(pid)) not in id_dict.keys():
                    id_dict[int(pid)] = []

                id_dict[int(pid)].append({'img_path': img_path, 'index': int(index), 'pid': int(pid), 'masks_path': masks_path,
                                          'camid': int(camid), 'team_id': int(team_id), 'gameid': int(game_id), 'role': role})

                data.append({'img_path': img_path,
				'index': int(index),
                                'pid': int(pid),
                                'masks_path': masks_path,
                                'camid': int(camid),
                                'team_id': team_id})

        if mode == 1:
            ids = list(set([i['pid'] for i in data]))
            index = np.random.permutation(len(ids))
            indice = [ids[j] for j in index]
            #indice = list(set([i['pid'] for i in data]))
            for player in indice:
                idx = np.random.permutation(len(id_dict[player]))
                data2 += [id_dict[player][f] for f in idx]

            for i, player in enumerate(data2):
                idx = indice.index(player['pid'])
                data2[i]['pid'] = idx

            return data2, indice

        elif mode == 2:
            ids = list(set([i['pid'] for i in data]))
            index = np.random.permutation(len(ids))
            indice = [ids[j] for j in index]
            for player in indice:
                idx = np.random.permutation(len(id_dict[player]))
                data2 += [id_dict[player][f] for f in idx]

            for i, player in enumerate(data2):
                idx = indice.index(player['pid'])
                data2[i]['pid'] = idx

            return data2, indice

        elif mode == 3:
            indice = list(set([i['pid'] for i in data]))
            indice = [i for i in indice if i in mapping]
            for player in indice:
                idx = np.random.permutation(len(id_dict[player]))
                data2 += [id_dict[player][f] for f in idx]

            for i, player in enumerate(data2):
                idx = mapping.index(player['pid'])
                data2[i]['pid'] = idx

            return data2, None

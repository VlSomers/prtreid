from __future__ import division, print_function, absolute_import

import re
import glob
import os.path as osp
import warnings
from pathlib import Path

import pandas as pd

from ..dataset import ImageDataset


class Dartfish(ImageDataset):
    """Dartfish
    file format: GID2013_LID51_MID4_CID1_FID1018_BID8_30x49.png


    #### Dataset preparation for ReID task

    **```Naming convention of image:```**
    ```GID{Global_tracklet_ID}LID{local_tracklet_ID}_MID{Match_ID}_CID{CameraID}_FID{Frame_ID}_BID{Bbox_index}_{width}x{height}.png```
    `

    **```example:```**
    ```GID45_LID12_MID4_CID3_FID26_BID5_66x67.png```

    ```
    Global_tracket_ID  : 45 (this is created by us for each new tracklet across cameras, games and splits). Range is [0 to ~3000].
    Local_tracklet_ID  : 12 (this is provided by dartfish). Ranges upto ~150
    Match_ID           : Game ID from {1,2,3,4,5}. Total 5 Games at present
    CameraID           : Camera ID {1,2,3,4}. Four cameras in total
    Frame_ID           : frame index in the given video {0-5000}. 5000 frames per video
    Bbox_index         : the nth instance in the given frame. At the maximum 16 players
    W                  : width of instance
    H                  : height of instance
    ```
    """
    _junk_pids = [0, -1]
    dataset_dir = 'dartfish_reid'
    masks_base_dir = 'masks'
    eval_metric = 'dartfish'

    masks_dirs = {

    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in Dartfish.masks_dirs:
            return None
        else:
            return masks_dir[masks_dir]

    def __init__(self,
                 root='',
                 masks_dir=None,
                 config=None,
                 **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.masks_dir = masks_dir
        self.train_size = config.dartfish.train_size
        self.val_size = config.dartfish.val_size
        self.test_size = config.dartfish.test_size
        self.query_per_id = config.dartfish.query_per_id

        # allow alternative directory structure
        if not osp.isdir(self.dataset_dir):
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        required_files = [
            self.dataset_dir, self.train_dir, self.val_dir, self.test_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        val = self.process_dir(self.val_dir, relabel=False)
        test = self.process_dir(self.test_dir, relabel=False)

        print("Reducing size of {} train ({}), val ({}) and test ({}) sets".format(self.__class__, len(train), len(val), len(test)))
        train = self.filter(train, self.train_size)
        val = self.filter(val, self.val_size)
        test = self.filter(test, self.test_size)

        query, gallery = self.query_gallery_split(test, self.query_per_id)

        if len(query) == 0 or len(gallery) == 0:
            raise RuntimeError("Dartfish query or gallery set shouldn't be empty")

        super(Dartfish, self).__init__(train, query, gallery, config=config, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')
        pattern = re.compile('([A-Z]+)([\d]+)_([A-Z]+)([-\d]+)_([A-Z]+)([-\d]+)_([A-Z]+)([-\d]+)_([A-Z]+)([-\d]+)_([A-Z]+)([-\d]+)_([-\d]+)x([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            metadata = self.get_metadata(img_path, pattern)
            pid = metadata['GID']
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            metadata = self.get_metadata(img_path, pattern)
            pid = metadata['GID']
            camid = metadata['CID']
            if pid == -1:
                continue # junk images are just ignored
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append({'img_path': img_path,
                         'pid': pid,
                         'camid': camid,
                         'local_id': metadata['LID'],
                         'game_id': metadata['MID'],
                         'frame_idx': metadata['FID'],
                         'bbox_idx': metadata['BID'],
                         'width': metadata['width'],
                         'height': metadata['height']
                         })
        return data

    def get_metadata(self, img_path, pattern):
        filename = Path(img_path).stem
        groups = pattern.search(filename).groups()
        metadata = {}
        for i in range(0, len(groups) - 2, 2):
            metadata[groups[i]] = int(groups[i + 1])
        metadata["width"] = groups[-2]
        metadata["height"] = groups[-1]
        return metadata

    def filter(self, set, reduce_size=10000):
        sampling_step = int(len(set) / reduce_size)
        df = pd.DataFrame.from_records(set)
        df = df.sort_values(by=['pid', 'frame_idx'])
        df = df[::sampling_step]
        # make sure pids are 0-based increasing numbers
        df.pid = pd.factorize(df.pid)[0]
        return df.to_dict('records')

    def query_gallery_split(self, set, query_per_id=2):
        df = pd.DataFrame.from_records(set)
        query = df.sort_values('frame_idx').groupby('pid').head(query_per_id)
        gallery = df.drop(query.index)
        return query.to_dict('records'), gallery.to_dict('records')

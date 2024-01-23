from __future__ import division, print_function, absolute_import
import glob
import os.path as osp

from ..dataset import ImageDataset


class EpflSport(ImageDataset):
    """EpflSport Dataset.
    """
    _junk_pids = [0, -1]
    dataset_dir = 'epfl_sport'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]

        self.check_before_run(required_files)

        train = []
        query = self.process_dir(self.query_dir, 2)
        gallery = self.process_dir(self.gallery_dir, 3)

        super(EpflSport, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, main_path, camid):
        data = []
        img_paths = glob.glob(osp.join(main_path, '*.png'))
        for pid, img_path in enumerate(img_paths):
            data.append({'img_path': img_path, 'pid': pid, 'camid': camid})

        return data

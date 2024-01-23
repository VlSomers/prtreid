from __future__ import division, print_function, absolute_import
import glob
import os.path as osp
import re

from ..dataset import ImageDataset


class DEChallengeSynergy(ImageDataset):
    """Synergy Dataset for data efficiency challenge at ICCV.
    """
    _junk_pids = [0, -1]
    dataset_dir = 'data_efficiency_challenge_synergy'
    dataset_url = None
    masks_base_dir = 'masks'
    masks_base_dir = 'masks'

    masks_dirs = {
        # dir_name: (masks_stack_size, contains_background_mask) # FIXME cannot use as target if no mask available
        'pifpaf': (36, False, '.jpg.confidence_fields.npy'),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in DEChallengeSynergy.masks_dirs:
            return None
        else:
            return DEChallengeSynergy.masks_dirs[masks_dir]

    def __init__(self, root='', masks_dir=None, **kwargs):
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        self.train_dir = osp.join(self.data_dir, 'training_val')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]

        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, 0, relabel=True)
        query = self.process_dir(self.query_dir, 1, relabel=False)
        gallery = self.process_dir(self.gallery_dir, 2, relabel=False)

        super(DEChallengeSynergy, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, camid, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpeg'))
        pattern = re.compile(r'([\d]+)_([\d]+)_([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid = pattern.search(img_path).groups()[0]
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid = pattern.search(img_path).groups()[0]
            # camid = int(pattern.search(img_path).groups()[2])
            if pid == -1:
                continue # junk images are just ignored
            if relabel:
                pid = pid2label[pid]
            # masks_path = self.infer_masks_path(img_path)
            data.append({'img_path': img_path,
                         'pid': pid,
                         # 'masks_path': masks_path,
                         'camid': camid})

        return data

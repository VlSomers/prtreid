from __future__ import division, print_function, absolute_import
import glob
import os.path as osp
import re

from ..dataset import ImageDataset


class SynergySequences(ImageDataset):
    """Synergy Dataset for data efficiency challenge at ICCV.
    """
    _junk_pids = [0, -1]
    dataset_dir = 'synergy_sequences_dataset'
    dataset_url = None
    masks_base_dir = 'masks'
    external_annotation_base_dir = 'external_annotation'

    masks_dirs = {
        # dir_name: (masks_stack_size, contains_background_mask)
        'pifpaf': (36, False, '.jpg.confidence_fields.npy'),
        'pifpaf_maskrcnn_filtering': (36, False, '.npy'),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in SynergySequences.masks_dirs:
            return None
        else:
            return SynergySequences.masks_dirs[masks_dir]

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
        self.train_dir = osp.join(self.data_dir, 'bbox_train_full')
        self.query_dir = osp.join(self.data_dir, 'bbox_val_full')
        self.gallery_dir = osp.join(self.data_dir, 'bbox_val_full')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]

        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir,is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(SynergySequences, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d{2})sSEQ-[A-Za-z]{2}-([0-9]+)')

        data = []
        for img_path in img_paths:
            pid, frame_idx, sequence_idx = map(int, pattern.search(img_path).groups())
            # pid += 1  # adding background
            if pid == -1:
                continue  # junk images are just ignored
            tot_pids = 3481 + 871 + 1
            assert 0 <= pid <= tot_pids  # pid == 0 means background
            assert 0 <= frame_idx <= 19  # index starts from 0
            # if is_train:
                # pid = self.dataset_dir + "_" + str(pid)
                # frame_idx = self.dataset_dir + "_" + str(frame_idx)
            masks_path = self.infer_masks_path(img_path)
            data.append({'img_path': img_path,
                         'pid': pid,
                         'masks_path': masks_path,
                         'camid': frame_idx})

        return data

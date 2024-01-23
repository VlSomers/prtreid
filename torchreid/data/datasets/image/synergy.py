from __future__ import division, print_function, absolute_import
import glob
import os.path as osp
import os

from ..dataset import ImageDataset


class Synergy(ImageDataset):
    """Synergy Dataset.
    """
    _junk_pids = [0, -1]
    dataset_dir = 'synergy_small'
    dataset_url = None
    masks_base_dir = 'masks'

    masks_dirs = {
        # dir_name: (masks_stack_size, contains_background_mask)
        'pifpaf': (36, False, '.confidence_fields.npy')
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in Synergy.masks_dirs:
            return None
        else:
            return Synergy.masks_dirs[masks_dir]

    def infer_masks_path(self, img_path): # FIXME remove when all datasets migrated
        masks_path = img_path + self.masks_suffix
        return masks_path

    def __init__(self, root='', masks_dir=None, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]

        self.check_before_run(required_files)

        train, _, _ = self.process_dir(self.train_dir, 1)
        query, seq2pid2label, previous_id_increment = self.process_dir(self.query_dir, 2)
        gallery, _, _ = self.process_dir(self.gallery_dir, 3, seq2pid2label, previous_id_increment)

        super(Synergy, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, main_path, camid, seq2pid2label=None, previous_id_increment=0):
        if seq2pid2label is None:
            seq2pid2label = {}
        sequences_dir_list = [f for f in os.scandir(main_path) if f.is_dir()]
        data = []

        for seq_dir in sequences_dir_list:
            seq_name = seq_dir.name
            dir_path = osp.join(main_path, seq_dir)
            pids_dir_list = [f for f in os.scandir(dir_path) if f.is_dir()]
            if seq_name not in seq2pid2label:
                pid2label = {pids_dir.name: label + previous_id_increment for label, pids_dir in enumerate(pids_dir_list)}
                for label, pids_dir in enumerate(pids_dir_list):
                    pid2label[pids_dir.name] = label + previous_id_increment
                seq2pid2label[seq_name] = pid2label
                previous_id_increment += len(pid2label)
            else:
                pid2label = seq2pid2label[seq_name]
                label = 0
                for pids_dir in pids_dir_list:
                    if pids_dir.name not in pid2label:
                        pid2label[pids_dir.name] = label + previous_id_increment
                        label += 1
                previous_id_increment += label

            for pid_dir in pids_dir_list:
                img_paths = glob.glob(osp.join(pid_dir.path, '*.png'))
                for img_path in img_paths:
                    pid = pid2label[pid_dir.name]
                    masks_path = self.infer_masks_path(img_path)
                    data.append({'img_path': img_path,
                                 'pid': pid,
                                 'masks_path': masks_path,
                                 'camid': camid})

        return data, seq2pid2label, previous_id_increment

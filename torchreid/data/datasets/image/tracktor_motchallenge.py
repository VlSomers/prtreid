from __future__ import absolute_import, division, print_function

import json
import os
import os.path as osp
import numpy as np
import pandas as pd
import tqdm
from ..dataset import ImageDataset

# Source: https://github.com/phil-bergmann/tracking_wo_bnw

def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


# def anns2df(anns, img_dir):
#     # Build DF from anns
#     to_kps = lambda x: np.array(x['keypoints']).reshape(-1, 3)
#     rows = []
#     for ann in tqdm.tqdm(anns['annotations']):
#         row={'path': f"{img_dir}/{ann['id']}.png",
#             'model_id': int(ann['model_id']),
#             'height': int(ann['bbox'][-1]),
#             'width': int(ann['bbox'][-2]),
#             'iscrowd': int(ann['iscrowd']),
#             'isnight': int(ann['is_night']),
#             'visibility' : (to_kps(ann)[..., 2] ==2).mean(),
#             'frame_n': int(ann['frame_n']),
#             **{f'attr_{i}': int(attr_val) for i, attr_val in enumerate(ann['attributes'])}}
#         rows.append(row)
#
#     return pd.DataFrame(rows)

# def assign_ids(df, night_id=True, attr_indices=None):
#     if attr_indices is None:
#         attr_indices = [0, 2, 3, 4, 7, 8, 9, 10]
#
#     id_cols = ['model_id'] + [f'attr_{i}' for i in attr_indices if f'attr{i}' in df.columns]
#     if night_id and 'isnight' in df.columns:
#         id_cols += ['isnight']
#
#     unique_ids_df = df[id_cols].drop_duplicates()
#     unique_ids_df['reid_id'] = np.arange(unique_ids_df.shape[0])
#
#     return df.merge(unique_ids_df)


def clean_rows(df, min_vis, min_h, min_w, min_samples, sample_nth_frame):
    # iscrowd: 1 if object must be ignored, else 0
    # vis: bbox confidence
    # Filter by size and occlusion
    if min_vis == 1.0:
        keep = (df['visibility'] >= min_vis) & (df['height']>=min_h) & (df['width'] >= min_w) & (df['iscrowd']==0)
    else:
        keep = (df['visibility'] > min_vis) & (df['height']>=min_h) & (df['width'] >= min_w) & (df['iscrowd']==0)

    clean_df = df[keep].copy()

    if sample_nth_frame:
        clean_df = clean_df.groupby('reid_id').apply(lambda _df: _df.iloc[::sample_nth_frame]).reset_index(drop=True)

    # Keep only ids with at least MIN_SAMPLES appearances
    clean_df['samples_per_id'] = clean_df.groupby('reid_id')['height'].transform('count').values
    clean_df = clean_df[clean_df['samples_per_id']>=min_samples]

    return clean_df


def relabel_ids(df):
    df.rename(columns = {'reid_id': 'reid_id_old'}, inplace=True)

    # Relabel Ids from 0 to N-1
    ids_df = df[['reid_id_old']].drop_duplicates()
    ids_df['reid_id'] = np.arange(ids_df.shape[0])
    df = df.merge(ids_df)
    return df


def to_tuple_list(df):
    return list(df[['path', 'reid_id', 'cam_id']].itertuples(index=False, name=None))


def sample_random_per_reid_id(df, num):
    per_reid_id = df.groupby('reid_id')['index'].agg(lambda x: list(np.random.choice(list(x.unique()), size=num, replace=False)))
    return per_reid_id.explode()


class TracktorMOTChallenge(ImageDataset):
    dataset_dir = 'MOTChallenge/MOT17'

    def __init__(self, seq_name, root, **kwargs):
        self.seq_name = seq_name
        self.min_vis = 0.0
        self.min_h = 50
        self.min_w = 25
        self.min_samples = 10
        self.sample_nth_frame = 0
        self.num_per_id_query = 5
        self.num_per_id_gallery = 1  # single-gallery-shot setting, where each gallery identity has only one instance

        root_dir = osp.join(osp.abspath(osp.expanduser(root)), self.dataset_dir)
        print(f"Preparing MOTSeqDataset dataset {seq_name} from {root_dir}.")

        df = self.get_dataframe(root_dir)
        train = to_tuple_list(df)

        # random single-shot query/gallery
        np.random.seed(0)
        query_per_id = sample_random_per_reid_id(df, self.num_per_id_query)
        query_df = df.loc[query_per_id.values].copy()
        gallery_df = df.drop(query_per_id).copy()

        gallery_per_id = sample_random_per_reid_id(gallery_df, self.num_per_id_gallery)
        gallery_df = gallery_df.loc[gallery_per_id.values].copy()

        # max frame distance single-shot query/gallery
        # query_per_id = df.groupby('reid_id')['index'].agg(lambda x: sorted(list(x.unique()))[0])
        # query_df = df.loc[query_per_id.values].copy()
        # gallery_df = df.drop(query_per_id).copy()

        # gallery_per_id = gallery_df.groupby('reid_id')['index'].agg(lambda x: sorted(list(x.unique()))[-1])
        # gallery_df = gallery_df.loc[gallery_per_id.values].copy()

        # IMPORTANT: For testing, torchreid only compares gallery and query images from different cam_ids
        # therefore we just assign them ids 0 and 1 respectively
        gallery_df['cam_id'] = 1

        query=to_tuple_list(query_df)
        gallery=to_tuple_list(gallery_df)

        super(TracktorMOTChallenge, self).__init__(train, query, gallery, **kwargs)

    def anns2df_motcha(self, anns, img_dir):
        # Build DF from anns
        rows = []
        for ann in tqdm.tqdm(anns['annotations']):
            box_path = osp.join(img_dir, str(self.seq_name), str(ann['ped_id']), f"{int(ann['frame_n'])}_{ann['id']}.png")
            row = {'path': box_path,
                   'ped_id': int(ann['ped_id']),
                   'height': int(ann['bbox'][-1]),
                   'width': int(ann['bbox'][-2]),
                   'iscrowd': int(ann['iscrowd']),
                   'visibility': float(ann['visibility']),
                   'frame_n': int(ann['frame_n'])}
            rows.append(row)

        return pd.DataFrame(rows)

    def get_dataframe(self, root_dir):
        ann_file = os.path.join(root_dir, 'anns', f'{self.seq_name}.json')
        img_dir = os.path.join(root_dir, 'imgs')

        # Create a Pandas DataFrame out of json annotations file
        anns = read_json(ann_file)

        df = self.anns2df_motcha(anns, img_dir)
        df['reid_id'] = df['ped_id']

        df = clean_rows(df,
            self.min_vis,
            min_h=self.min_h,
            min_w=self.min_w,
            min_samples=self.min_samples,
            sample_nth_frame=self.sample_nth_frame)
        df = relabel_ids(df)
        df['cam_id'] = 0

        df['index'] = df.index.values

        return df


def get_sequence_class(seq_name):

    dataset_class = TracktorMOTChallenge

    class MOTSeqDatasetWrapper(dataset_class):
        def __init__(self, **kwargs):
            super(MOTSeqDatasetWrapper, self).__init__(seq_name, **kwargs)

    MOTSeqDatasetWrapper.__name__ = seq_name

    return MOTSeqDatasetWrapper

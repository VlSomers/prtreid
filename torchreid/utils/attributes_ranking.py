import numpy as np
from mat4py import loadmat

from torchreid import metrics


def display_attributes_ranking(body_parts_distmat, distmat, g_camids, g_pids, q_camids, q_pids, use_metric_cuhk03):
    # Abandonned: attributes annotation are not precise enough to rank parts according to their attribute appearance
    bp_attributes_mapping = [
        [],
        ["upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray", "upblue", "upgreen"],
        ["upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray", "upblue", "upgreen"],
        ["downblack", "downwhite", "downpink", "downpurple", "downyellow", "downgray", "downblue", "downgreen",
         "downbrown"],
        [],
    ]

    bp_attributes = {}

    # attributes_file_path = '/Users/vladimirsomers/datasets/reid/market1501/Market-1501_Attribute-master/market_attribute.mat'
    attributes_file_path = '/home/vso/datasets/reid/market1501/Market-1501-v15.09.15/market_attribute.mat'
    attributes_dataset = loadmat(
        attributes_file_path)

    test_attributes = attributes_dataset['market_attribute']['test']

    # TODO NOT exactly THE SAME ids as Occluded duke????
    # bp_attributes_mapping = [
    #     [],
    #     ['upblack', 'upwhite', 'upred', 'uppurple', 'upgray', 'upblue', 'upgreen', 'upbrown'],
    #     ['upblack', 'upwhite', 'upred', 'uppurple', 'upgray', 'upblue', 'upgreen', 'upbrown'],
    #     ['downblack', 'downwhite', 'downred', 'downgray', 'downblue', 'downgreen', 'downbrown'],
    #     [],
    # ]
    #
    #
    # bp_attributes = {}
    #
    #
    # attributes_file_path = '/home/vso/datasets/reid/Occluded_Duke/duke_attribute.mat'
    # attributes_dataset = loadmat(attributes_file_path)
    #
    # test_attributes = attributes_dataset['duke_attribute']['test']

    for bp, attributes in enumerate(bp_attributes_mapping):
        if len(attributes) != 0:
            labels = []
            for attribute in attributes:
                labels.append(np.array(test_attributes[attribute]))
            labels = np.array(labels)
            # assert (labels-1).sum(axis=0).max() == 1 # TODO false for DUKE
            merged_labels = labels.argmax(0)
            bp_attributes[bp] = np.concatenate(([merged_labels.max() + 1], merged_labels))

    # pids_index = [int(entry) for i, entry in enumerate(test_attributes['image_index'])]
    mapping = {}
    for i, entry in enumerate(test_attributes['image_index']):
        mapping[int(entry)] = i + 1

    mapping[0] = 0

    q_indexes = []
    for pid in q_pids:
        q_indexes.append(mapping[pid])

    g_indexes = []
    for pid in g_pids:
        g_indexes.append(mapping[pid])

    for bp in range(0, body_parts_distmat.shape[0]):  # TODO DO NOT TAKE INTO ACCOUNT -1 DISTANCES!!!!
        # if attribute mat = pids as rows and attributes as columns, return columns corresponding to target body part attribute
        # for each dataset and each transform, need a mapping from body part number to attribute idx
        # [int(os.path.basename(entry[0])[0:4]) == entry[1] for entry in self.data]
        if bp in bp_attributes:
            cmc, mAP = metrics.evaluate_rank(
                body_parts_distmat[bp],
                bp_attributes[bp][q_indexes],
                bp_attributes[bp][g_indexes],
                q_camids,
                g_camids,
                use_metric_cuhk03=use_metric_cuhk03
            )
            print(
                'For BP {}, mAP: {:.2%} and Rank-1: {:.2%} and Rank-5: {:.2%} and Rank-10: {:.2%} and Rank-50: {:.2%}'.format(
                    bp, mAP, cmc[0], cmc[4], cmc[9], cmc[49]))
    distmat[:,
    g_pids == 0] = distmat.max()  # +1% map on six_body_masks, identity and full body, +6% with 36bm and intra_id_mean

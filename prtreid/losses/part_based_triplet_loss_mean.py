from __future__ import division, absolute_import

from prtreid.losses.part_based_triplet_loss import PartBasedTripletLoss
from prtreid.utils.tensortools import masked_mean


class PartBasedTripletLossMean(PartBasedTripletLoss):

    def __init__(self, **kwargs):
        super(PartBasedTripletLossMean, self).__init__(**kwargs)

    def _combine_body_parts_dist_matrices(self, per_body_part_pairwise_dist, valid_body_part_pairwise_dist_mask, labels):
        if valid_body_part_pairwise_dist_mask is not None:
            self.writer.update_invalid_body_part_pairwise_distances_count(valid_body_part_pairwise_dist_mask)
            pairwise_dist = masked_mean(per_body_part_pairwise_dist, valid_body_part_pairwise_dist_mask)
        else:
            valid_body_part_pairwise_dist = per_body_part_pairwise_dist
            pairwise_dist = valid_body_part_pairwise_dist.mean(0)

        return pairwise_dist

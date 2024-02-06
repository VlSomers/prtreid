from __future__ import division, absolute_import

from prtreid.losses.part_based_triplet_loss import PartBasedTripletLoss
from prtreid.utils.tensortools import replace_values


class IntraPartsTripletLoss(PartBasedTripletLoss):

    def __init__(self, **kwargs):
        super(IntraPartsTripletLoss, self).__init__(**kwargs)

    def _combine_body_parts_dist_matrices(self, per_body_part_pairwise_dist, valid_body_part_pairwise_dist_mask, labels):
        if valid_body_part_pairwise_dist_mask is not None:
            valid_body_part_pairwise_dist = replace_values(per_body_part_pairwise_dist, ~valid_body_part_pairwise_dist_mask, -1)
            self.writer.update_invalid_body_part_pairwise_distances_count(valid_body_part_pairwise_dist_mask)
        else:
            valid_body_part_pairwise_dist = per_body_part_pairwise_dist

        return valid_body_part_pairwise_dist

from __future__ import division, absolute_import

import torch

from prtreid.losses.part_based_triplet_loss import PartBasedTripletLoss
from prtreid.utils.tensortools import replace_values


class PartBasedTripletLossMin(PartBasedTripletLoss):

    def __init__(self, **kwargs):
        super(PartBasedTripletLossMin, self).__init__(**kwargs)

    def _combine_body_parts_dist_matrices(self, per_body_part_pairwise_dist, valid_body_part_pairwise_dist_mask, labels):
        if valid_body_part_pairwise_dist_mask is not None:
            max_value = torch.finfo(per_body_part_pairwise_dist.dtype).max
            valid_body_part_pairwise_dist = replace_values(per_body_part_pairwise_dist, ~valid_body_part_pairwise_dist_mask, max_value)
            self.writer.update_invalid_body_part_pairwise_distances_count(valid_body_part_pairwise_dist_mask)
        else:
            valid_body_part_pairwise_dist = per_body_part_pairwise_dist

        pairwise_dist, body_part_id = valid_body_part_pairwise_dist.min(0)

        if valid_body_part_pairwise_dist_mask is not None:
            invalid_pairwise_dist_mask = valid_body_part_pairwise_dist_mask.sum(dim=0) == 0
            pairwise_dist = replace_values(pairwise_dist, invalid_pairwise_dist_mask, -1)

        body_parts_count = per_body_part_pairwise_dist.shape[0] # body parts masks count
        if per_body_part_pairwise_dist.shape[0] > 1:
            self.writer.used_body_parts_statistics(body_parts_count, body_part_id)

        return pairwise_dist

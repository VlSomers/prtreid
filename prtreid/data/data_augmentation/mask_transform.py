from __future__ import division, print_function, absolute_import

import torch
from torch import nn
import numpy as np
from albumentations import (
    DualTransform
)
import torch.nn.functional as F

# FIXME better implementation, remove duplicate code

class MaskTransform(DualTransform):
    def __init__(self):
        super(MaskTransform, self).__init__(always_apply=True, p=1)

    def apply(self, img, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError("Method apply_to_keypoint is not implemented in class " + self.__class__.__name__)


class PermuteMasksDim(MaskTransform):
    def apply_to_mask(self, masks, **params):
        return masks.permute(2, 0, 1)


class ResizeMasks(MaskTransform):
    def __init__(self, height, width, mask_scale):
        super(ResizeMasks, self).__init__()
        self._size = (int(height/mask_scale), int(width/mask_scale))

    def apply_to_mask(self, masks, **params):
        return nn.functional.interpolate(masks.unsqueeze(0), self._size, mode='nearest').squeeze(0)  # Best perf with nearest here and bilinear in parts engine


class RemoveBackgroundMask(MaskTransform):
    def apply_to_mask(self, masks, **params):
        return masks[:, :, 1::]


class AddBackgroundMask(MaskTransform):
    def __init__(self, background_computation_strategy='sum', softmax_weight=0, mask_filtering_threshold=0.3):
        super().__init__()
        self.background_computation_strategy = background_computation_strategy
        self.softmax_weight = softmax_weight
        self.mask_filtering_threshold = mask_filtering_threshold

    def apply_to_mask(self, masks, **params):
        if self.background_computation_strategy == 'sum':
            background_mask = 1 - masks.sum(dim=0)
            masks = torch.cat([background_mask.unsqueeze(0), masks])
        elif self.background_computation_strategy == 'threshold':
            background_mask = masks.max(dim=0)[0] < self.mask_filtering_threshold
            masks = torch.cat([background_mask.unsqueeze(0), masks])
        elif self.background_computation_strategy == 'diff_from_max':
            background_mask = 1 - masks.max(dim=0)[0]
            masks = torch.cat([background_mask.unsqueeze(0), masks])
        else:
            raise ValueError('Background mask combine strategy {} not supported'.format(self.background_computation_strategy))
        if self.softmax_weight > 0:
            masks = F.softmax(masks * self.softmax_weight, dim=0)
        return masks


class CombinePifPafIntoFullBodyMask(MaskTransform):
    parts_names = ['full_combined']
    parts_num = 1
    def apply_to_mask(self, masks, **params):
        # return torch.max(masks, 0, keepdim=True)[0]
        # +- 1% improvement in mAP by using sum instead of max
        # return torch.sum(masks, 0, keepdim=True)
        return torch.clamp(torch.sum(masks, 0, keepdim=True), 0, 1)


class IdentityMask(MaskTransform):
    parts_names = ['id']
    parts_num = 1
    def apply_to_mask(self, masks, **params):
        return torch.ones((1, masks.shape[1], masks.shape[2]))


class PCBMasks2(MaskTransform):
    parts_num = 2
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        self._size = masks.shape[1:3]
        self.stripe_height = self._size[0] / self.parts_num

        self.pcb_masks = torch.zeros((self.parts_num, self._size[0], self._size[1]))

        stripes_range = np.round(np.arange(0, self.parts_num + 1) * self._size[0] / self.parts_num).astype(int)
        for i in range(0, stripes_range.size-1):
            self.pcb_masks[i, stripes_range[i]:stripes_range[i+1], :] = 1

        return self.pcb_masks


class PCBMasks3(MaskTransform):
    parts_num = 3
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        self._size = masks.shape[1:3]
        self.stripe_height = self._size[0] / self.parts_num

        self.pcb_masks = torch.zeros((self.parts_num, self._size[0], self._size[1]))

        stripes_range = np.round(np.arange(0, self.parts_num + 1) * self._size[0] / self.parts_num).astype(int)
        for i in range(0, stripes_range.size-1):
            self.pcb_masks[i, stripes_range[i]:stripes_range[i+1], :] = 1

        return self.pcb_masks


class PCBMasks4(MaskTransform):
    parts_num = 4
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        self._size = masks.shape[1:3]
        self.stripe_height = self._size[0] / self.parts_num

        self.pcb_masks = torch.zeros((self.parts_num, self._size[0], self._size[1]))

        stripes_range = np.round(np.arange(0, self.parts_num + 1) * self._size[0] / self.parts_num).astype(int)
        for i in range(0, stripes_range.size-1):
            self.pcb_masks[i, stripes_range[i]:stripes_range[i+1], :] = 1

        return self.pcb_masks


class PCBMasks5(MaskTransform):
    parts_num = 5
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        self._size = masks.shape[1:3]
        self.stripe_height = self._size[0] / self.parts_num

        self.pcb_masks = torch.zeros((self.parts_num, self._size[0], self._size[1]))

        stripes_range = np.round(np.arange(0, self.parts_num + 1) * self._size[0] / self.parts_num).astype(int)
        for i in range(0, stripes_range.size-1):
            self.pcb_masks[i, stripes_range[i]:stripes_range[i+1], :] = 1

        return self.pcb_masks


class PCBMasks6(MaskTransform):
    parts_num = 6
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        self._size = masks.shape[1:3]
        self.stripe_height = self._size[0] / self.parts_num

        self.pcb_masks = torch.zeros((self.parts_num, self._size[0], self._size[1]))

        stripes_range = np.round(np.arange(0, self.parts_num + 1) * self._size[0] / self.parts_num).astype(int)
        for i in range(0, stripes_range.size-1):
            self.pcb_masks[i, stripes_range[i]:stripes_range[i+1], :] = 1

        return self.pcb_masks


class PCBMasks7(MaskTransform):
    parts_num = 7
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        self._size = masks.shape[1:3]
        self.stripe_height = self._size[0] / self.parts_num

        self.pcb_masks = torch.zeros((self.parts_num, self._size[0], self._size[1]))

        stripes_range = np.round(np.arange(0, self.parts_num + 1) * self._size[0] / self.parts_num).astype(int)
        for i in range(0, stripes_range.size-1):
            self.pcb_masks[i, stripes_range[i]:stripes_range[i+1], :] = 1

        return self.pcb_masks


class PCBMasks8(MaskTransform):
    parts_num = 8
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        self._size = masks.shape[1:3]
        self.stripe_height = self._size[0] / self.parts_num

        self.pcb_masks = torch.zeros((self.parts_num, self._size[0], self._size[1]))

        stripes_range = np.round(np.arange(0, self.parts_num + 1) * self._size[0] / self.parts_num).astype(int)
        for i in range(0, stripes_range.size-1):
            self.pcb_masks[i, stripes_range[i]:stripes_range[i+1], :] = 1

        return self.pcb_masks


class AddFullBodyMaskToBaseMasks(MaskTransform):
    parts_num = 37
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        full_body_mask = torch.max(masks, 0, keepdim=True)[0]

        return torch.cat([  masks,
                            full_body_mask
                          ])


class AddFullBodyMaskAndFullBoundingBoxToBaseMasks(MaskTransform):
    parts_num = 38
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        full_body_mask = torch.max(masks, 0, keepdim=True)[0]

        full_bounding_box = torch.ones(masks.shape[1:3]).unsqueeze(0)

        return torch.cat([  masks,
                            full_body_mask,
                            full_bounding_box
                          ])


class CombinePifPafIntoMultiScaleBodyMasks(MaskTransform):
    parts_num = 9
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs'random_flip'
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0].unsqueeze(0)
        arms_mask = torch.max(torch.cat([left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist,
                                         right_wrist, left_shoulder_to_left_elbow, right_shoulder_to_right_elbow,
                                         left_elbow_to_left_wrist, right_elbow_to_right_wrist]), 0)[0].unsqueeze(0)
        torso_mask = torch.max(torch.cat([left_shoulder, right_shoulder, left_hip, right_hip, left_hip_to_right_hip,
                                          left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)[0].unsqueeze(0)
        legs_mask = torch.max(torch.cat([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle,
                                         left_ankle_to_left_knee, left_knee_to_left_hip, right_ankle_to_right_knee,
                                         right_knee_to_right_hip, left_hip_to_right_hip]), 0)[0].unsqueeze(0)
        feet_mask = torch.max(torch.cat([left_ankle, right_ankle]), 0)[0].unsqueeze(0)

        upper_body = torch.max(torch.cat([torso_mask, arms_mask, head_mask]), 0)[0].unsqueeze(0)

        lower_body = torch.max(torch.cat([legs_mask, feet_mask]), 0)[0].unsqueeze(0)

        full_body_mask = torch.max(masks, 0, keepdim=True)[0]

        return torch.cat([  masks,
                            head_mask,
                            torso_mask,
                            arms_mask,
                            legs_mask,
                            feet_mask,
                            upper_body,
                            lower_body,
                            full_body_mask
                          ])

class CombinePifPafIntoOneBodyMasks(MaskTransform):
    parts_names = ['full']
    parts_num = 1

    def apply_to_mask(self, masks, **params):
        full_body_mask = torch.max(masks, 0, keepdim=True)[0]
        return full_body_mask

class CombinePifPafIntoTwoBodyMasks(MaskTransform):
    parts_num = 2
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        torso_arms_head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder, left_shoulder, right_shoulder,
                                               left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                               left_shoulder_to_right_shoulder,
                                               left_elbow, right_elbow, left_wrist,
                                               right_wrist, left_shoulder_to_left_elbow, right_shoulder_to_right_elbow,
                                               left_elbow_to_left_wrist, right_elbow_to_right_wrist
                                               ]), 0)[0]
        legs_mask = torch.max(torch.cat([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle,
                                         left_ankle_to_left_knee, left_knee_to_left_hip, right_ankle_to_right_knee,
                                         right_knee_to_right_hip, left_hip_to_right_hip]), 0)[0]

        return torch.cat([torso_arms_head_mask.unsqueeze(0), legs_mask.unsqueeze(0)])


class CombinePifPafIntoThreeBodyMasks(MaskTransform):
    parts_num = 3
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]
        torso_arms_mask = torch.max(torch.cat([left_shoulder, right_shoulder,
                                               left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                               left_shoulder_to_right_shoulder,
                                               left_elbow, right_elbow, left_wrist,
                                               right_wrist, left_shoulder_to_left_elbow, right_shoulder_to_right_elbow,
                                               left_elbow_to_left_wrist, right_elbow_to_right_wrist
                                               ]), 0)[0]
        legs_mask = torch.max(torch.cat([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle,
                                         left_ankle_to_left_knee, left_knee_to_left_hip, right_ankle_to_right_knee,
                                         right_knee_to_right_hip, left_hip_to_right_hip]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0), torso_arms_mask.unsqueeze(0), legs_mask.unsqueeze(0)])


class CombinePifPafIntoFourBodyMasks(MaskTransform):
    part_names = ["head", "torso", "arms", "legs"]
    parts_num = 4

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]
        arms_mask = torch.max(torch.cat([left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist,
                                         right_wrist, left_shoulder_to_left_elbow, right_shoulder_to_right_elbow,
                                         left_elbow_to_left_wrist, right_elbow_to_right_wrist]), 0)[0]
        torso_mask = torch.max(torch.cat([left_shoulder, right_shoulder, left_hip, right_hip, left_hip_to_right_hip,
                                          left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)[0]
        legs_mask = torch.max(torch.cat([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle,
                                         left_ankle_to_left_knee, left_knee_to_left_hip, right_ankle_to_right_knee,
                                         right_knee_to_right_hip, left_hip_to_right_hip]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0), torso_mask.unsqueeze(0), arms_mask.unsqueeze(0),
                          legs_mask.unsqueeze(0)])


class CombinePifPafIntoFourBodyMasksNoOverlap(MaskTransform):
    parts_num = 4
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]
        arms_mask = torch.max(torch.cat([left_elbow, right_elbow, left_wrist,
                                         right_wrist, left_shoulder_to_left_elbow, right_shoulder_to_right_elbow,
                                         left_elbow_to_left_wrist, right_elbow_to_right_wrist]), 0)[0]
        torso_mask = torch.max(torch.cat([left_shoulder, right_shoulder, left_hip, right_hip, left_hip_to_right_hip,
                                          left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)[0]
        legs_mask = torch.max(torch.cat([left_knee, right_knee, left_ankle, right_ankle,
                                         left_ankle_to_left_knee, left_knee_to_left_hip, right_ankle_to_right_knee,
                                         right_knee_to_right_hip, left_hip_to_right_hip]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0), torso_mask.unsqueeze(0), arms_mask.unsqueeze(0),
                          legs_mask.unsqueeze(0)])

        # head_mask = torch.sum(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
        #                                  nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
        #                                  right_eye_to_right_ear, left_ear_to_left_shoulder,
        #                                  right_ear_to_right_shoulder]), 0)
        # arms_mask = torch.sum(torch.cat([left_elbow, right_elbow, left_wrist,
        #                                  right_wrist, left_shoulder_to_left_elbow, right_shoulder_to_right_elbow,
        #                                  left_elbow_to_left_wrist, right_elbow_to_right_wrist]), 0)
        # torso_mask = torch.sum(torch.cat([left_shoulder, right_shoulder, left_hip, right_hip, left_hip_to_right_hip,
        #                                   left_shoulder_to_left_hip, right_shoulder_to_right_hip,
        #                                   left_shoulder_to_right_shoulder]), 0)
        # legs_mask = torch.sum(torch.cat([left_knee, right_knee, left_ankle, right_ankle,
        #                                  left_ankle_to_left_knee, left_knee_to_left_hip, right_ankle_to_right_knee,
        #                                  right_knee_to_right_hip, left_hip_to_right_hip]), 0)
        #
        # return torch.cat([head_mask.unsqueeze(0), torso_mask.unsqueeze(0), arms_mask.unsqueeze(0),
        #                   legs_mask.unsqueeze(0)])

class CombinePifPafIntoFourVerticalParts(MaskTransform):
    parts_num = 4
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]

        arms_torso_mask = torch.max(torch.cat([left_elbow, right_elbow, left_wrist,
                                         right_wrist, left_shoulder_to_left_elbow, right_shoulder_to_right_elbow,
                                         left_elbow_to_left_wrist, right_elbow_to_right_wrist, left_shoulder, right_shoulder, left_hip, right_hip, left_hip_to_right_hip,
                                          left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)[0]

        legs_mask = torch.max(torch.cat([left_hip, right_hip, left_knee, right_knee,
                                         left_ankle_to_left_knee, left_knee_to_left_hip, right_ankle_to_right_knee,
                                         right_knee_to_right_hip]), 0)[0]
        feet_mask = torch.max(torch.cat([left_ankle, right_ankle]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0), arms_torso_mask.unsqueeze(0), legs_mask.unsqueeze(0), feet_mask.unsqueeze(0)])


class CombinePifPafIntoFourVerticalPartsPif(MaskTransform):
    parts_num = 4
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear]), 0)[0]

        arms_torso_mask = torch.max(torch.cat([left_elbow, right_elbow, left_wrist, right_wrist, left_shoulder, right_shoulder, left_hip, right_hip]), 0)[0]

        legs_mask = torch.max(torch.cat([left_hip, right_hip, left_knee, right_knee]), 0)[0]
        feet_mask = torch.max(torch.cat([left_ankle, right_ankle]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0), arms_torso_mask.unsqueeze(0), legs_mask.unsqueeze(0), feet_mask.unsqueeze(0)])


class CombinePifPafIntoFiveVerticalParts(MaskTransform):
    parts_names = ["head", "upper_torso", "lower_torso", "legs", "feet"]
    parts_num = 5

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]

        upper_arms_torso_mask = torch.max(torch.cat([left_elbow, right_elbow,
                                                     left_shoulder_to_left_elbow, right_shoulder_to_right_elbow,
                                                     left_shoulder, right_shoulder,
                                                     left_shoulder_to_right_shoulder]), 0)[0]

        lower_arms_torso_mask = torch.max(torch.cat([left_wrist, right_wrist,
                                                     left_elbow_to_left_wrist, right_elbow_to_right_wrist,
                                                     left_hip, right_hip,
                                                     right_shoulder_to_right_hip]), 0)[0]

        legs_mask = torch.max(torch.cat([left_hip, right_hip, left_knee, right_knee,
                                         left_ankle_to_left_knee, left_knee_to_left_hip, right_ankle_to_right_knee,
                                         right_knee_to_right_hip]), 0)[0]

        feet_mask = torch.max(torch.cat([left_ankle, right_ankle]), 0)[0]


        return torch.cat([head_mask.unsqueeze(0), upper_arms_torso_mask.unsqueeze(0),  lower_arms_torso_mask.unsqueeze(0), legs_mask.unsqueeze(0), feet_mask.unsqueeze(0)])


class CombinePifPafIntoFiveBodyMasks(MaskTransform):
    parts_names = ["head", "upper_torso", "lower_torso", "legs", "feet"]
    parts_num = 5

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]

        arms_mask = torch.max(torch.cat([left_shoulder, left_elbow, left_wrist, left_shoulder_to_left_elbow,
                                             left_elbow_to_left_wrist, right_shoulder, right_elbow, right_wrist, right_shoulder_to_right_elbow,
                                              right_elbow_to_right_wrist]), 0)[0]

        torso_mask = torch.max(torch.cat([left_hip, right_hip, left_hip_to_right_hip,
                                          left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)[0]

        legs_mask = torch.max(torch.cat([left_hip_to_right_hip, left_hip, right_hip, left_knee, right_knee,
                                         left_ankle_to_left_knee, left_knee_to_left_hip, right_ankle_to_right_knee,
                                         right_knee_to_right_hip]), 0)[0]

        feet_mask = torch.max(torch.cat([left_ankle, right_ankle]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0), torso_mask.unsqueeze(0), arms_mask.unsqueeze(0),
                          legs_mask.unsqueeze(0), feet_mask.unsqueeze(0)])


class CombinePifPafIntoSixVerticalParts(MaskTransform):
    parts_num = 6
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]


        arms_mask = torch.sum(torch.cat([left_shoulder, left_elbow, left_wrist, left_shoulder_to_left_elbow,
                                             left_elbow_to_left_wrist, right_shoulder, right_elbow, right_wrist, right_shoulder_to_right_elbow,
                                              right_elbow_to_right_wrist]), 0)

        # arms_mask = torch.sum(torch.cat([left_elbow, left_wrist, left_shoulder_to_left_elbow,
        #                                      left_elbow_to_left_wrist]), 0)

        upper_torso_mask = torch.sum(torch.cat([left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)

        lower_torso_mask = torch.sum(torch.cat([left_hip, right_hip, left_hip_to_right_hip]), 0)

        legs_mask = torch.max(torch.cat([left_hip, right_hip, left_knee, right_knee,
                                         left_ankle_to_left_knee, left_knee_to_left_hip, right_ankle_to_right_knee,
                                         right_knee_to_right_hip]), 0)[0]

        feet_mask = torch.max(torch.cat([left_ankle, right_ankle]), 0)[0]


        return torch.cat([head_mask.unsqueeze(0), arms_mask.unsqueeze(0), upper_torso_mask.unsqueeze(0), lower_torso_mask.unsqueeze(0), legs_mask.unsqueeze(0), feet_mask.unsqueeze(0)])


class CombinePifPafIntoSixBodyMasks(MaskTransform):
    parts_num = 6
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]
        left_arm_mask = torch.max(torch.cat([left_shoulder, left_elbow, left_wrist, left_shoulder_to_left_elbow,
                                             left_elbow_to_left_wrist]), 0)[0]
        right_arm_mask = torch.max(torch.cat([right_shoulder, right_elbow, right_wrist, right_shoulder_to_right_elbow,
                                              right_elbow_to_right_wrist]), 0)[0]
        torso_mask = torch.max(torch.cat([left_hip, right_hip, left_hip_to_right_hip,
                                          left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)[0]
        left_leg_mask = torch.max(torch.cat([left_knee, left_ankle, left_ankle_to_left_knee,
                                             left_knee_to_left_hip, left_hip_to_right_hip]), 0)[0]
        right_leg_mask = torch.max(torch.cat([right_knee, right_ankle, right_ankle_to_right_knee,
                                         right_knee_to_right_hip]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0), torso_mask.unsqueeze(0), left_arm_mask.unsqueeze(0),
                          right_arm_mask.unsqueeze(0), left_leg_mask.unsqueeze(0), right_leg_mask.unsqueeze(0)])


class CombinePifPafIntoSixBodyMasksSum(MaskTransform):
    parts_num = 6
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        # # TODO normalize or clip to [0, 1] ?
        # +- 1% improvement in mAP by using sum instead of max
        head_mask = torch.sum(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)
        left_arm_mask = torch.sum(torch.cat([left_shoulder, left_elbow, left_wrist, left_shoulder_to_left_elbow,
                                             left_elbow_to_left_wrist]), 0)
        right_arm_mask = torch.sum(torch.cat([right_shoulder, right_elbow, right_wrist, right_shoulder_to_right_elbow,
                                              right_elbow_to_right_wrist]), 0)
        torso_mask = torch.sum(torch.cat([left_hip, right_hip, left_hip_to_right_hip,
                                          left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)
        left_leg_mask = torch.sum(torch.cat([left_knee, left_ankle, left_ankle_to_left_knee,
                                             left_knee_to_left_hip, left_hip_to_right_hip]), 0)
        right_leg_mask = torch.sum(torch.cat([right_knee, right_ankle, right_ankle_to_right_knee,
                                         right_knee_to_right_hip]), 0)

        return torch.clamp(torch.cat([head_mask.unsqueeze(0), torso_mask.unsqueeze(0), left_arm_mask.unsqueeze(0),
                          right_arm_mask.unsqueeze(0), left_leg_mask.unsqueeze(0), right_leg_mask.unsqueeze(0)]), 0, 1)


class CombinePifPafIntoSixBodyMasksSimilarToEight(MaskTransform):
    parts_num = 6
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]
        torso_mask = torch.max(torch.cat([left_hip, right_hip, left_hip_to_right_hip,
                                          left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)[0]
        left_arm_mask = torch.max(torch.cat([left_shoulder, left_elbow, left_wrist, left_shoulder_to_left_elbow,
                                             left_elbow_to_left_wrist]), 0)[0]
        right_arm_mask = torch.max(torch.cat([right_shoulder, right_elbow, right_wrist, right_shoulder_to_right_elbow,
                                              right_elbow_to_right_wrist]), 0)[0]
        leg_mask = torch.max(torch.cat([left_knee, left_ankle_to_left_knee,
                                             left_knee_to_left_hip, left_hip_to_right_hip, right_knee, right_ankle_to_right_knee,
                                         right_knee_to_right_hip]), 0)[0]
        feet_mask = torch.max(torch.cat([left_ankle, right_ankle]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0), torso_mask.unsqueeze(0), left_arm_mask.unsqueeze(0),
                          right_arm_mask.unsqueeze(0), leg_mask.unsqueeze(0), feet_mask.unsqueeze(0)])


class CombinePifPafIntoEightBodyMasks(MaskTransform):
    parts_num = 8
    parts_names = ['head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg', 'left_feet', 'right_feet']

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]
        left_arm_mask = torch.max(torch.cat([left_shoulder, left_elbow, left_wrist, left_shoulder_to_left_elbow,
                                             left_elbow_to_left_wrist]), 0)[0]
        right_arm_mask = torch.max(torch.cat([right_shoulder, right_elbow, right_wrist, right_shoulder_to_right_elbow,
                                              right_elbow_to_right_wrist]), 0)[0]
        torso_mask = torch.max(torch.cat([left_hip, right_hip, left_hip_to_right_hip,
                                          left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)[0]
        left_leg_mask = torch.max(torch.cat([left_knee, left_ankle_to_left_knee,
                                             left_knee_to_left_hip, left_hip_to_right_hip]), 0)[0]
        right_leg_mask = torch.max(torch.cat([right_knee, right_ankle_to_right_knee,
                                         right_knee_to_right_hip]), 0)[0]
        left_feet_mask = torch.max(torch.cat([left_ankle]), 0)[0]
        right_feet_mask = torch.max(torch.cat([right_ankle]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0), torso_mask.unsqueeze(0), left_arm_mask.unsqueeze(0),
                          right_arm_mask.unsqueeze(0), left_leg_mask.unsqueeze(0), right_leg_mask.unsqueeze(0),
                          left_feet_mask.unsqueeze(0), right_feet_mask.unsqueeze(0)])


class CombinePifPafIntoEightVerticalBodyMasks(MaskTransform):
    parts_num = 8
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]
        left_arm_mask = torch.max(torch.cat([left_shoulder, left_elbow, left_wrist, left_shoulder_to_left_elbow,
                                             left_elbow_to_left_wrist]), 0)[0]
        right_arm_mask = torch.max(torch.cat([right_shoulder, right_elbow, right_wrist, right_shoulder_to_right_elbow,
                                              right_elbow_to_right_wrist]), 0)[0]
        torso_mask = torch.max(torch.cat([left_hip, right_hip, left_hip_to_right_hip,
                                          left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)[0]
        left_leg_mask = torch.max(torch.cat([left_knee, left_ankle_to_left_knee,
                                             left_knee_to_left_hip, left_hip_to_right_hip]), 0)[0]
        right_leg_mask = torch.max(torch.cat([right_knee, right_ankle_to_right_knee,
                                         right_knee_to_right_hip]), 0)[0]
        left_feet_mask = torch.max(torch.cat([left_ankle]), 0)[0]
        right_feet_mask = torch.max(torch.cat([right_ankle]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0), torso_mask.unsqueeze(0), left_arm_mask.unsqueeze(0),
                          right_arm_mask.unsqueeze(0), left_leg_mask.unsqueeze(0), right_leg_mask.unsqueeze(0),
                          left_feet_mask.unsqueeze(0), right_feet_mask.unsqueeze(0)])


class CombinePifPafIntoTenMSBodyMasks(MaskTransform):
    parts_num = 10
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]
        left_arm_mask = torch.max(torch.cat([left_shoulder, left_elbow, left_wrist, left_shoulder_to_left_elbow,
                                             left_elbow_to_left_wrist]), 0)[0]
        right_arm_mask = torch.max(torch.cat([right_shoulder, right_elbow, right_wrist, right_shoulder_to_right_elbow,
                                              right_elbow_to_right_wrist]), 0)[0]
        torso_mask = torch.max(torch.cat([left_hip, right_hip, left_hip_to_right_hip,
                                          left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)[0]
        left_leg_mask = torch.max(torch.cat([left_knee, left_ankle_to_left_knee,
                                             left_knee_to_left_hip, left_hip_to_right_hip]), 0)[0]
        right_leg_mask = torch.max(torch.cat([right_knee, right_ankle_to_right_knee,
                                         right_knee_to_right_hip]), 0)[0]
        left_feet_mask = torch.max(torch.cat([left_ankle]), 0)[0]
        right_feet_mask = torch.max(torch.cat([right_ankle]), 0)[0]

        upper_body_mask = torch.max(torch.cat([head_mask.unsqueeze(0), left_arm_mask.unsqueeze(0), right_arm_mask.unsqueeze(0), torso_mask.unsqueeze(0)]), 0)[0]
        lower_body_mask = torch.max(torch.cat([left_leg_mask.unsqueeze(0), right_leg_mask.unsqueeze(0), left_feet_mask.unsqueeze(0), right_feet_mask.unsqueeze(0)]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0), torso_mask.unsqueeze(0), left_arm_mask.unsqueeze(0),
                          right_arm_mask.unsqueeze(0), left_leg_mask.unsqueeze(0), right_leg_mask.unsqueeze(0),
                          left_feet_mask.unsqueeze(0), right_feet_mask.unsqueeze(0),
                          upper_body_mask.unsqueeze(0), lower_body_mask.unsqueeze(0),
                          ])

class CombinePifPafIntoSevenVerticalBodyMasks(MaskTransform):
    parts_num = 7
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]
        shoulders_mask = torch.max(torch.cat([left_shoulder, right_shoulder, left_shoulder_to_right_shoulder]), 0)[0]
        elbow_mask =  torch.max(torch.cat([left_elbow, right_elbow]), 0)[0]
        wrist_mask = torch.max(torch.cat([left_wrist, right_wrist]), 0)[0]
        hip_mask = torch.max(torch.cat([left_hip, right_hip, left_hip_to_right_hip]), 0)[0]
        knee_mask = torch.max(torch.cat([left_knee, right_knee]), 0)[0]
        ankle_mask = torch.max(torch.cat([left_ankle, right_ankle]), 0)[0]

        # torch.max(torch.cat([]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0),
                            shoulders_mask.unsqueeze(0),
                            elbow_mask.unsqueeze(0),
                            wrist_mask.unsqueeze(0),
                            hip_mask.unsqueeze(0),
                            knee_mask.unsqueeze(0),
                            ankle_mask.unsqueeze(0),
                          ])


class CombinePifPafIntoSevenBodyMasksSimilarToEight(MaskTransform):
    parts_num = 7
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]
        left_arm_mask = torch.max(torch.cat([left_shoulder, left_elbow, left_wrist, left_shoulder_to_left_elbow,
                                             left_elbow_to_left_wrist]), 0)[0]
        right_arm_mask = torch.max(torch.cat([right_shoulder, right_elbow, right_wrist, right_shoulder_to_right_elbow,
                                              right_elbow_to_right_wrist]), 0)[0]
        upper_torso_mask = torch.max(torch.cat([left_shoulder_to_left_hip, right_shoulder_to_right_hip,
                                          left_shoulder_to_right_shoulder]), 0)[0]
        lower_torso_mask = torch.max(torch.cat([left_hip, right_hip, left_hip_to_right_hip]), 0)[0]
        leg_mask = torch.max(torch.cat([left_knee, left_ankle_to_left_knee,
                                             left_knee_to_left_hip, left_hip_to_right_hip, right_knee, right_ankle_to_right_knee,
                                         right_knee_to_right_hip]), 0)[0]
        feet_mask = torch.max(torch.cat([left_ankle, right_ankle]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0), upper_torso_mask.unsqueeze(0), lower_torso_mask.unsqueeze(0), left_arm_mask.unsqueeze(0),
                          right_arm_mask.unsqueeze(0), leg_mask.unsqueeze(0), feet_mask.unsqueeze(0)])



class CombinePifPafIntoElevenBodyMasks(MaskTransform):
    parts_num = 11
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear, left_ear_to_left_shoulder,
                                         right_ear_to_right_shoulder]), 0)[0]
        left_elbow_mask = torch.max(torch.cat([left_shoulder, left_elbow, left_shoulder_to_left_elbow]), 0)[0]
        left_wrist_mask = torch.max(torch.cat([left_wrist, left_elbow_to_left_wrist]), 0)[0]

        right_elbow_mask = torch.max(torch.cat([right_shoulder, right_elbow, right_shoulder_to_right_elbow]), 0)[0]
        right_wrist_mask = torch.max(torch.cat([right_wrist, right_elbow_to_right_wrist]), 0)[0]

        upper_torso_mask = torch.max(torch.cat([left_shoulder_to_left_hip, right_shoulder_to_right_hip, left_shoulder_to_right_shoulder]), 0)[0]

        lower_torso_mask = torch.max(torch.cat([left_hip, right_hip, left_hip_to_right_hip]), 0)[0]
        left_leg_mask = torch.max(torch.cat([left_knee, left_knee_to_left_hip, left_hip_to_right_hip]), 0)[0]
        right_leg_mask = torch.max(torch.cat([right_knee, right_knee_to_right_hip]), 0)[0]
        left_feet_mask = torch.max(torch.cat([left_ankle_to_left_knee, left_ankle]), 0)[0]
        right_feet_mask = torch.max(torch.cat([right_ankle_to_right_knee, right_ankle]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0),
                          left_elbow_mask.unsqueeze(0), left_wrist_mask.unsqueeze(0),
                          right_elbow_mask.unsqueeze(0), right_wrist_mask.unsqueeze(0),
                          upper_torso_mask.unsqueeze(0), lower_torso_mask.unsqueeze(0),
                          left_leg_mask.unsqueeze(0), right_leg_mask.unsqueeze(0),
                          left_feet_mask.unsqueeze(0), right_feet_mask.unsqueeze(0)])


class CombinePifPafIntoFourteenBodyMasks(MaskTransform):
    parts_num = 14
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        nose = masks[0].unsqueeze(0) # head
        left_eye = masks[1].unsqueeze(0) # head
        right_eye = masks[2].unsqueeze(0) # head
        left_ear = masks[3].unsqueeze(0) # head
        right_ear = masks[4].unsqueeze(0) # head
        left_shoulder = masks[5].unsqueeze(0) # arms - torso
        right_shoulder = masks[6].unsqueeze(0) # arms - torso
        left_elbow = masks[7].unsqueeze(0) # arms
        right_elbow = masks[8].unsqueeze(0) # arms
        left_wrist = masks[9].unsqueeze(0) # arms
        right_wrist = masks[10].unsqueeze(0) # arms
        left_hip = masks[11].unsqueeze(0) # torso - legs
        right_hip = masks[12].unsqueeze(0) # torso - legs
        left_knee = masks[13].unsqueeze(0) # legs
        right_knee = masks[14].unsqueeze(0) # legs
        left_ankle = masks[15].unsqueeze(0) # legs
        right_ankle = masks[16].unsqueeze(0) # legs
        left_ankle_to_left_knee = masks[17].unsqueeze(0) # legs
        left_knee_to_left_hip = masks[18].unsqueeze(0) # legs
        right_ankle_to_right_knee = masks[19].unsqueeze(0) # legs
        right_knee_to_right_hip = masks[20].unsqueeze(0) # legs
        left_hip_to_right_hip = masks[21].unsqueeze(0) # legs - torso
        left_shoulder_to_left_hip = masks[22].unsqueeze(0) # torso
        right_shoulder_to_right_hip = masks[23].unsqueeze(0) # torso
        left_shoulder_to_right_shoulder = masks[24].unsqueeze(0) # torso
        left_shoulder_to_left_elbow = masks[25].unsqueeze(0) # arms
        right_shoulder_to_right_elbow = masks[26].unsqueeze(0) # arms
        left_elbow_to_left_wrist = masks[27].unsqueeze(0) # arms
        right_elbow_to_right_wrist = masks[28].unsqueeze(0) # arms
        left_eye_to_right_eye = masks[29].unsqueeze(0) # head
        nose_to_left_eye = masks[30].unsqueeze(0) # head
        nose_to_right_eye = masks[31].unsqueeze(0) # head
        left_eye_to_left_ear = masks[32].unsqueeze(0) # head
        right_eye_to_right_ear = masks[33].unsqueeze(0) # head
        left_ear_to_left_shoulder = masks[34].unsqueeze(0) # head
        right_ear_to_right_shoulder = masks[35].unsqueeze(0) # head

        head_mask = torch.max(torch.cat([nose, left_eye, right_eye, left_ear, right_ear, left_eye_to_right_eye,
                                         nose_to_left_eye, nose_to_right_eye, left_eye_to_left_ear,
                                         right_eye_to_right_ear]), 0)[0]
        neck_mask = torch.max(torch.cat([left_ear_to_left_shoulder, right_ear_to_right_shoulder]), 0)[0]
        left_elbow_mask = torch.max(torch.cat([left_shoulder, left_elbow, left_shoulder_to_left_elbow]), 0)[0]
        left_wrist_mask = torch.max(torch.cat([left_wrist, left_elbow_to_left_wrist]), 0)[0]

        right_elbow_mask = torch.max(torch.cat([right_shoulder, right_elbow, right_shoulder_to_right_elbow]), 0)[0]
        right_wrist_mask = torch.max(torch.cat([right_wrist, right_elbow_to_right_wrist]), 0)[0]

        upper_torso_mask = torch.max(torch.cat([left_shoulder_to_left_hip, right_shoulder_to_right_hip, left_shoulder_to_right_shoulder]), 0)[0]

        lower_torso_mask = torch.max(torch.cat([left_hip, right_hip, left_hip_to_right_hip]), 0)[0]
        left_leg_mask = torch.max(torch.cat([left_knee, left_knee_to_left_hip, left_hip_to_right_hip]), 0)[0]
        right_leg_mask = torch.max(torch.cat([right_knee, right_knee_to_right_hip]), 0)[0]
        left_tibia_mask = torch.max(torch.cat([left_ankle_to_left_knee]), 0)[0]
        right_tibia_mask = torch.max(torch.cat([right_ankle_to_right_knee]), 0)[0]
        left_feet_mask = torch.max(torch.cat([left_ankle]), 0)[0]
        right_feet_mask = torch.max(torch.cat([right_ankle]), 0)[0]

        return torch.cat([head_mask.unsqueeze(0),
                          neck_mask.unsqueeze(0),
                          left_elbow_mask.unsqueeze(0), left_wrist_mask.unsqueeze(0),
                          right_elbow_mask.unsqueeze(0), right_wrist_mask.unsqueeze(0),
                          upper_torso_mask.unsqueeze(0), lower_torso_mask.unsqueeze(0),
                          left_leg_mask.unsqueeze(0), right_leg_mask.unsqueeze(0),
                          left_tibia_mask.unsqueeze(0), right_tibia_mask.unsqueeze(0),
                          left_feet_mask.unsqueeze(0), right_feet_mask.unsqueeze(0)])

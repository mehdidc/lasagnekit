"""
Generator that augments data in realtime
"""

import numpy as np
import skimage
import skimage.transform


def fast_warp(img, tf, output_shape=(53, 53), mode='constant'):
    """
    This wrapper function is about five times faster than skimage.transform.warp, for our use case.
    """
    m = tf._matrix
    if len(output_shape) == 2:
        output_shape_ = output_shape + (1,)
        img = img.reshape(output_shape_)
    else:
        output_shape_ = output_shape
    img_wf = np.empty(output_shape_, dtype='float32')
    c = output_shape_[-1]
    for k in range(c):
        img_wf[:, :, k] = skimage.transform._warps_cy._warp_fast(
                img[:, :, k],
                m,
                output_shape=output_shape_[0:-1], mode=mode)
    return img_wf.reshape(output_shape)


def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0),
                                 h=32, w=32):

    center_shift = np.array((h, w)) / 2. - 0.5
    tform_center = skimage.transform.SimilarityTransform(
        translation=-center_shift)
    tform_uncenter = skimage.transform.SimilarityTransform(
        translation=center_shift)

    tform_augment = skimage.transform.AffineTransform(scale=(
        1 / zoom, 1 / zoom), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    # shift to center, augment, shift back (for the rotation/shearing)
    tform = tform_center + tform_augment + tform_uncenter

    return tform


def random_perturbation_transform(rng, zoom_range, rotation_range, shear_range, translation_range, do_flip=False, w=32, h=32):
    shift_x = rng.uniform(*translation_range)
    shift_y = rng.uniform(*translation_range)
    translation = (shift_x, shift_y)

    # random rotation [0, 360]
    # there is no post-augmentation, so full rotations here!
    rotation = rng.uniform(*rotation_range)

    # random shear [0, 5]
    shear = rng.uniform(low=shear_range[0], high=shear_range[1])
    # # flip
    if do_flip and (rng.randint(2) > 0):  # flip half of the time
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    # random zoom [0.9, 1.1]
    # zoom = np.random.uniform(*zoom_range)
    log_zoom_range = [np.log(z) for z in zoom_range]
    # for a zoom factor this sampling approach makes more sense.
    zoom = np.exp(rng.uniform(*log_zoom_range))
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead
    # of [0.9, 1.1] makes more sense.
    return build_augmentation_transform(zoom, rotation, shear, translation, w=w, h=h)

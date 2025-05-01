import nibabel 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from segmentation import *
import itertools
import nibabel
from skimage.segmentation import find_boundaries
from segmentation import my_seg  # or however you import your custom seg



# def process_file(file_list,index,myseg):
#     # Take for argument the name of both segmentations, in the right order ED then ES !
#     df = pd.DataFrame(columns=["Id","ED_RV_volume","ED_LV_volume","ED_MY_volume","ES_RV_volume","ES_LV_volume","ES_MY_volume"])
#     for ind,file_name in enumerate(file_list) :
#         seg_nii = nibabel.load(file_name)
#         seg_data = np.asanyarray(seg_nii.dataobj, dtype=np.uint8)
#         if myseg : 
#             print("utilisation de ma segmentation")
#             seg_data = my_seg(seg_data)
#         labels, counts = np.unique(seg_data, return_counts=True)

#         for label, count in zip(labels, counts):
#             if ind ==0 :
#                 if label==1:
#                     ED_RV_volume = count
#                 elif label==3 : 
#                     ED_LV_volume = count
#                 elif label ==2:
#                     ED_MY_volume = count
#             elif ind ==1 :
#                 if label==1:
#                     ES_RV_volume = count
#                 elif label==3 : 
#                     ES_LV_volume = count
#                 elif label ==2:
#                     ES_MY_volume = count
#     df.loc[len(df)] = [int(index),ED_RV_volume,ED_LV_volume,ED_MY_volume,ES_RV_volume,ES_LV_volume,ES_MY_volume]   
#     # df["RV_DIFF"] = abs(df["ED_RV_volume"] - df["ES_RV_volume"])
#     # df["LV_DIFF"] = abs(df["ED_LV_volume"] - df["ES_LV_volume"])
#     # df["MY_DIFF"] = abs(df["ED_MY_volume"] - df["ES_MY_volume"])
#     return df


def compute_volume_features(folder_path,index,myseg = False):
    
    file_segED = str(index) + "_ED_SEG.nii"

    file_segES = str(index) + "_ES_SEG.nii"

    DIR_SEGED = os.path.join(folder_path,file_segED)
    DIR_SEGES = os.path.join(folder_path,file_segES)
    df = process_file_bis([DIR_SEGED,DIR_SEGES],index,myseg)
    return df   


def count_segmentation_border(mask, label=None, connectivity=1, mode='outer'):
    """
    Count border voxels of `label` in `mask`.
    """
    # build a binary mask of the region
    if label is None:
        fg = mask != 0
    else:
        fg = (mask == label)
    # find boundary pixels
    border = find_boundaries(fg, connectivity=connectivity, mode=mode)
    return int(np.count_nonzero(border))

def process_file_bis(file_list, index, myseg=False):
    """
    For a pair of (ED, ES) segmentation files, compute:
      - volumes (voxel counts) for RV=1, LV=3, MYO=2
      - border counts for each of those labels
    Returns a one-row DataFrame with columns:
      Id,
      ED_RV_vol, ED_RV_border,
      ED_LV_vol, ED_LV_border,
      ED_MY_vol, ED_MY_border,
      ES_RV_vol, ES_RV_border,
      ES_LV_vol, ES_LV_border,
      ES_MY_vol, ES_MY_border
    """
    cols = [
        "Id",
        "ED_RV_vol",   "ED_RV_border",
        "ED_LV_vol",   "ED_LV_border",
        "ED_MY_vol",   "ED_MY_border",
        "ES_RV_vol",   "ES_RV_border",
        "ES_LV_vol",   "ES_LV_border",
        "ES_MY_vol",   "ES_MY_border",
    ]
    df = pd.DataFrame(columns=cols)

    # placeholders
    rec = {c: 0 for c in cols}
    rec["Id"] = int(index)

    for phase_idx, file_name in enumerate(file_list):
        # load segmentation
        img      = nibabel.load(file_name)
        seg_data = np.asanyarray(img.dataobj, dtype=np.uint8)
        if myseg:
            seg_data = my_seg(seg_data)

        # for each label, count voxels and border voxels
        for lbl, short in [(1, "RV"), (3, "LV"), (2, "MY")]:
            count = int((seg_data == lbl).sum())
            border_count = count_segmentation_border(seg_data, label=lbl)

            prefix = "ED" if phase_idx == 0 else "ES"
            rec[f"{prefix}_{short}_vol"]    = count
            rec[f"{prefix}_{short}_border"] = border_count

    # append to DataFrame
    df.loc[len(df)] = rec
    return df



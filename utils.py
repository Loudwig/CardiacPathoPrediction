import nibabel 
import os
import numpy as np
import pandas as pd


def process_file(file_list,index):
    # Take for argument the name of both segmentations, in the right order ED then ES !
    df = pd.DataFrame(columns=["Id","ED_RV_volume","ED_LV_volume","ED_MY_volume","ES_RV_volume","ES_LV_volume","ES_MY_volume"])
    for ind,file_name in enumerate(file_list) :
        seg_nii = nibabel.load(file_name)
        seg_data = np.asanyarray(seg_nii.dataobj, dtype=np.uint8)

        labels, counts = np.unique(seg_data, return_counts=True)

        for label, count in zip(labels, counts):
            if ind ==0 :
                if label==1:
                    ED_RV_volume = count
                elif label==3 : 
                    ED_LV_volume = count
                elif label ==2:
                    ED_MY_volume = count
            elif ind ==1 :
                if label==1:
                    ES_RV_volume = count
                elif label==3 : 
                    ES_LV_volume = count
                elif label ==2:
                    ES_MY_volume = count
    df.loc[len(df)] = [int(index),ED_RV_volume,ED_LV_volume,ED_MY_volume,ES_RV_volume,ES_LV_volume,ES_MY_volume]   
    df["RV_DIFF"] = abs(df["ED_RV_volume"] - df["ES_RV_volume"])
    df["LV_DIFF"] = abs(df["ED_LV_volume"] - df["ES_LV_volume"])
    df["MY_DIFF"] = abs(df["ED_MY_volume"] - df["ES_MY_volume"])
    return df


def compute_features(folder_path,index,test = False):
    
    file_segED = str(index) + "_ED_SEG.nii"

    file_segES = str(index) + "_ES_SEG.nii"

    DIR_SEGED = os.path.join(folder_path,file_segED)
    DIR_SEGES = os.path.join(folder_path,file_segES)
    df = process_file([DIR_SEGED,DIR_SEGES],index)
    return df   

# a = process_file(["/Users/rplanchon/Documents/telecom/IMA/S2/IMA205/Challenge/CardiacPathoPrediction/Dataset/Train/001/001_ED_seg.nii","/Users/rplanchon/Documents/telecom/IMA/S2/IMA205/Challenge/CardiacPathoPrediction/Dataset/Train/001/001_ES_seg.nii"],"001")
# print(a)
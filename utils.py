import nibabel 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from segmentation import *
import itertools

def process_file(file_list,index,myseg):
    # Take for argument the name of both segmentations, in the right order ED then ES !
    df = pd.DataFrame(columns=["Id","ED_RV_volume","ED_LV_volume","ED_MY_volume","ES_RV_volume","ES_LV_volume","ES_MY_volume"])
    for ind,file_name in enumerate(file_list) :
        seg_nii = nibabel.load(file_name)
        seg_data = np.asanyarray(seg_nii.dataobj, dtype=np.uint8)
        if myseg : 
            print("utilisation de ma segmentation")
            seg_data = my_seg(seg_data)
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
    # df["RV_DIFF"] = abs(df["ED_RV_volume"] - df["ES_RV_volume"])
    # df["LV_DIFF"] = abs(df["ED_LV_volume"] - df["ES_LV_volume"])
    # df["MY_DIFF"] = abs(df["ED_MY_volume"] - df["ES_MY_volume"])
    return df


def compute_volume_features(folder_path,index,myseg = False):
    
    file_segED = str(index) + "_ED_SEG.nii"

    file_segES = str(index) + "_ES_SEG.nii"

    DIR_SEGED = os.path.join(folder_path,file_segED)
    DIR_SEGES = os.path.join(folder_path,file_segES)
    df = process_file([DIR_SEGED,DIR_SEGES],index,myseg)
    return df   


def compute_body_surface_area(height,weight): 
    # This is a formula I found online to compute body_surface area
    return 0.007184 * (height**0.725 )* (weight**0.425)

def add_body_surface_area_feature(df : pd.DataFrame ,name_column_height = "Height",name_column_weight = "Weight"):
    if (name_column_height and name_column_weight in df.columns) and ("body_surface" not in df.columns)  :
        df["body_surface"] = compute_body_surface_area(df[name_column_height],df[name_column_weight])
        print("body surface are feature added modified")
    else : 
        print("please provide a dataframe with a height and weight feature")
    
    
def add_ratio_features(df:pd.DataFrame):
    # This function will compute all the possible ratios.. 
    # Pour chaque paire de colonnes (A, B), on calcule A / B
    for col1, col2 in itertools.product(df.columns, repeat=2):
        if col1 != col2:
            new_col_name = f"{col1}_div_{col2}"
            # Attention à la division par zéro
            df[new_col_name] = df[col1] / df[col2].replace(0, float('nan'))

def augment_data(X, noise_factor=0.01):
    # Function that augment the data set by adding noising data

    # Compute the standard deviation of each feature
    std_devs = X.std(axis=0)
    
    # Add Gaussian noise to each feature based on its standard deviation
    noise = np.random.normal(loc=0, scale=noise_factor * std_devs, size=X.shape)
    
    # Create the noised dataset
    X_noisy = X + noise
    
    return X_noisy


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class GaussianNoiseInjector(BaseEstimator, TransformerMixin):
    """
    Injects Gaussian noise into features, proportionally to each feature's std-dev.
    
    Parameters
    ----------
    noise_factor : float, default=0.01
        Scale of the noise as a fraction of each feature’s standard deviation.
    random_state : int or None, default=None
        Seed for reproducible noise.
    """
    def __init__(self, noise_factor=0.01, random_state=None):
        self.noise_factor = noise_factor
        self.random_state = random_state

    def fit(self, X, y=None):
        # nothing to learn, but store RNG
        self._rng = np.random.RandomState(self.random_state)
        return self

    def transform(self, X):
        # work on numpy arrays (or convert DataFrame→array, then back)
        is_df = hasattr(X, "values")
        if is_df:
            cols = X.columns
            idx  = X.index
            arr  = X.values
        else:
            arr = np.asarray(X)
        
        # compute std per feature
        stds = arr.std(axis=0)
        # sample noise
        noise = self._rng.normal(
            loc=0,
            scale=self.noise_factor * stds,
            size=arr.shape
        )
        X_noisy = arr + noise
        
        if is_df:
            return type(X)(X_noisy, columns=cols, index=idx)
        else:
            return X_noisy

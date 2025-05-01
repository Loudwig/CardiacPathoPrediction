import itertools
import re
import pandas as pd
import numpy as np   # for safe division
import os

BASE       = os.getcwd()
OUT_DIR    = os.path.join(BASE, "data")
TRAIN_DIR = os.path.join(OUT_DIR,"Training_shape.csv")
TEST_DIR = os.path.join(OUT_DIR,"Testing_shape.csv")
df_train = pd.read_csv(TRAIN_DIR)        
df_test = pd.read_csv(TEST_DIR)

def add_ratios(df,Train=True):
    
    # --- 1 · isolate volume columns ---------------------------------------------
    vol_cols = [c for c in df.columns if re.search(r'MeshVolume$', c)]
    # e.g. ['ED_LV_shape_Volume', 'ES_LV_shape_Volume', 'ED_RV_shape_Volume', …]
    print(vol_cols)
    # --- 2 · pair-wise ratios with itertools.combinations -----------------------
    ratio_frames = {}          # will hold Series objects keyed by new column name

    for a, b in itertools.combinations(vol_cols, 2):
        name = f"{a}_over_{b}"              # ED_LV_shape_Volume_over_ES_LV_shape_Volume
        ratio = np.divide(df[a], df[b])
        ratio_frames[name] = ratio

    # merge them into the table
    df = pd.concat([df, pd.DataFrame(ratio_frames)], axis=1)

    return df
 

def compute_body_surface_area(height,weight): 
    "Return the body surface area from height and weight. (Formula of Du Bois)"
    
    return 0.007184 * (height**0.725 )* (weight**0.425)

def add_body_surface_area_feature(df : pd.DataFrame ,name_column_height = "Height",name_column_weight = "Weight"):
    # Description :
    # Add for each row the BSA associated.
    
    if (name_column_height and name_column_weight in df.columns) and ("body_surface" not in df.columns)  :
        df["body_surface"] = compute_body_surface_area(df[name_column_height],df[name_column_weight])
        #print("Body surface area feature added")
    else : 
        print("provide a dataframe with a height and weight feature")
       
def add_meta_data(df,Train= True):  
    if Train : 
        metaData = pd.read_csv(os.path.join(BASE,"Dataset","metaDataTrain.csv"))
    else :
        metaData = pd.read_csv(os.path.join(BASE,"Dataset","metaDataTest.csv")) 

    columns = ["Id","Height","Weight"]
    to_add = metaData[columns]
    df = df.merge(to_add,on="Id",how="right")
    add_body_surface_area_feature(df)           
    df.drop(columns=["Height", "Weight"], errors="ignore", inplace=True)
    
    return df
    

def main(df,Train=True):
    df = add_ratios(df,Train=Train)
    print("ratio added")
    df = add_meta_data(df,Train=Train)
    print("meta data added")

    if Train : 
        df.to_csv(os.path.join(OUT_DIR,"Trainning_full.csv"), index=False)
    else : 
        df.to_csv(os.path.join(OUT_DIR,"Testing_full.csv"), index=False)
        
main(df_train)
main(df_test,Train=False)
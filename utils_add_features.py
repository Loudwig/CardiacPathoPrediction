import numpy as np
import itertools
import pandas as pd

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
    
    
def add_ratio_features(df:pd.DataFrame,columns_to_exclude : list[str]=None):
    "Add ratios of a set of features. Choix de ces features seront détaillés dans le rapport."
    
    if columns_to_exclude is None:
        columns_to_exclude = []
    
    # Take only the columns of the df that are not exluded
    cols_to_compute = [
        col for col in df.columns
        if col not in columns_to_exclude
    ] 
    
    # Pour chaque combinaison (A, B), on calcule A / B.
    # On ne calcule pas A/B et B/A. Ce choix sera détaillé dans le rapport
    for col1, col2 in itertools.combinations(cols_to_compute, 2):
        if col1 != col2:
            new_col_name = f"{col1}_div_{col2}"
            # division par zéro
            df[new_col_name] = df[col1] / df[col2].replace(0, float('nan'))

def add_specific_ratio(df,col1,col2) : 
    "add a single ratio between two columns."
    
    if col1 and col2 in df.columns : 
        new_col_name = f"{col1}_div_{col2}"
        df[new_col_name] =  df[col1] / df[col2].replace(0, float('nan'))
    else : 
        print("columns not in dataframe")
                    

def clean_features(df: pd.DataFrame) -> pd.DataFrame:

    """
    Utilize the function above to augment the orignal dataset with other features.
    Must be use on X and X_test so they share the same features.
    
    Steps
    -----
    1. add body-surface area and drop raw Height and Weight
    2. scale every column (except 'body_surface') by BSA
    3. add generic & specific organ-to-border ratios
    4. drop raw border areas that are now redundant
    """
    df = df.copy()
    
    # 1) BSA 
    add_body_surface_area_feature(df)           
    df.drop(columns=["Height", "Weight"], errors="ignore", inplace=True)

    # 2) divide by BSA 
    numeric_cols = df.columns.drop("body_surface", errors="ignore")
    df[numeric_cols] = df[numeric_cols].div(df["body_surface"], axis=0)

    # 3) ratios 
    borders = [
        "ED_RV_border", "ED_LV_border", "ED_MY_border",
        "ES_RV_border", "ES_LV_border", "ES_MY_border"
    ]
    add_ratio_features(df, columns_to_exclude=borders + ["body_surface"])

    for num, den in [
        ("ED_RV_vol", "ED_RV_border"),
        ("ED_LV_vol", "ED_LV_border"),
        ("ED_MY_vol", "ED_MY_border"),
        ("ES_RV_vol", "ES_RV_border"),
        ("ES_LV_vol", "ES_LV_border"),
        ("ES_MY_vol", "ES_MY_border"),
    ]:
        add_specific_ratio(df, num, den)

    # 4) drop raw borders
    df.drop(columns=borders, errors="ignore", inplace=True)
    
    #print("Finished adding feature\n")
    return df

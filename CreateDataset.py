# Create Dataset with features
import os
import pandas as pd
from utils import * 

BASE_DIR = os.getcwd()
TRAIN_DIR = os.path.join(BASE_DIR,"Dataset/Train") 
TEST_DIR = os.path.join(BASE_DIR,"Dataset/Test") 
MYSEG = True
print(BASE_DIR,TRAIN_DIR,TEST_DIR)


# pour tout les folders : garder le nom comme ID, calculer les features pour chaque image
# df = pd.DataFrame(columns=["Id",  "ED_RV_volume",  "ED_LV_volume",  "ED_MY_volume",  "ES_RV_volume"  ,"ES_LV_volume"  ,"ES_MY_volume",  "RV_DIFF",  "LV_DIFF",  "MY_DIFF"])

def CreateDataset(Train = True) :
    
    if not MYSEG and not Train: 
        print("You must activate my seg if you want to create testing dataset")
    else : 
        df = pd.DataFrame(columns=["Id",  "ED_RV_volume",  "ED_LV_volume",  "ED_MY_volume",  "ES_RV_volume"  ,"ES_LV_volume"  ,"ES_MY_volume"])

        if Train : 
            
            for x in os.listdir(TRAIN_DIR): 
                folder_path = os.path.join(TRAIN_DIR,x)
                
                if os.path.isdir(folder_path):
                    new_row = compute_volume_features(folder_path,x,MYSEG)
                    df = pd.concat([df, new_row], ignore_index=True)

            # now add meta data  Height and Weight
            metaData = pd.read_csv("./Dataset/metaDataTrain.csv")
            columns = ["Id","Height","Weight"]
            to_add = metaData[columns]
            df = df.merge(to_add,on="Id",how = "right")
            df.drop(columns=df.columns[0],axis=1,inplace=True) # remove the first columns that is just another Id like column probably comes from the merge

            output_col = ["Id","Category"]
            output_df = metaData[output_col]
            output_df.drop(columns=output_df.columns[0],axis=1,inplace=True)

            if MYSEG : 
                output_df.to_csv("TrainningOutput_Dataset_myseg.csv")
                # save as csv the trainning dataset.
                df.to_csv("TrainningInput_Dataset_myseg.csv")
            else : 
                output_df.to_csv("TrainningOutput_Dataset.csv")
                # save as csv the trainning dataset.
                df.to_csv("TrainningInput_Dataset.csv")
        else : 
            for x in os.listdir(TEST_DIR): 
                folder_path = os.path.join(TEST_DIR,x)
                
                if os.path.isdir(folder_path):
                    new_row = compute_volume_features(folder_path,x,MYSEG)
                    df = pd.concat([df, new_row], ignore_index=True)

            # add meta data  Height and Weight
            metaData = pd.read_csv("./Dataset/metaDataTest.csv")
            columns = ["Id","Height","Weight"]
            to_add = metaData[columns]
            df = df.merge(to_add,on="Id",how = "right")
            df.drop(columns=df.columns[0],axis=1,inplace=True) # remove the first columns that is just another Id like column probably comes from the merge

            if MYSEG : 
                # save as csv the trainning dataset.
                df.to_csv("TestingInput_Dataset_myseg.csv")

CreateDataset(Train = False)
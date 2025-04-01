# Create Dataset with features
import os
import pandas as pd
from utils import * 

BASE_DIR = os.getcwd()
TRAIN_DIR = os.path.join(BASE_DIR,"Dataset/Train") 
TEST_DIR = os.path.join(BASE_DIR,"Dataset/Test") 
print(BASE_DIR,TRAIN_DIR,TEST_DIR)


# pour tout les folders : garder le nom comme ID, calculer les features pour chaque image
df = pd.DataFrame(columns=["Id",  "ED_RV_volume",  "ED_LV_volume",  "ED_MY_volume",  "ES_RV_volume"  ,"ES_LV_volume"  ,"ES_MY_volume",  "RV_DIFF",  "LV_DIFF",  "MY_DIFF"])

for x in os.listdir(TRAIN_DIR): 
    folder_path = os.path.join(TRAIN_DIR,x)
    
    if os.path.isdir(folder_path):
        new_row = compute_features(folder_path,x)
        df = pd.concat([df, new_row], ignore_index=True)

# now add meta data  Height and Weight
metaData = pd.read_csv("./Dataset/metaDataTrain.csv")
columns = ["Id","Height","Weight"]
to_add = metaData[columns]
df = df.merge(to_add,on="Id",how = "right")

output_col = ["Id","Category"]
output_df = metaData[output_col]
output_df.to_csv("TrainningOutput_Dataset.csv")
# save as csv the trainning dataset.
df.to_csv("TrainningInput_Dataset.csv")
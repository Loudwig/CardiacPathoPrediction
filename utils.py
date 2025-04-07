import nibabel 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from scipy.ndimage import binary_fill_holes


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
            print(label,count)
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

def flood_fill(matrix, seed, new_label, boundary_label1=2,boundary_label2=0):
    x, y = seed
    # If the seed is already a boundary or already the new label, nothing to do.
    if matrix[x, y] == boundary_label1 or boundary_label2 or matrix[x, y] == new_label:
        return matrix

    # The label to be replaced (target label)
    target_label = matrix[x, y]
    q = deque([seed])

    # Process the queue until empty
    while q:
        i, j = q.popleft()
        # Only fill if the current cell is the target label.
        if matrix[i, j] == target_label:
            matrix[i, j] = new_label

            # Check 4-connected neighbors (up, down, left, right)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                # Ensure we don't go out-of-bounds
                if 0 <= ni < matrix.shape[0] and 0 <= nj < matrix.shape[1]:
                    # Only add the neighbor if it’s not the boundary and still has the target label.
                    if matrix[ni, nj] != boundary_label1 or boundary_label2 and matrix[ni, nj] == target_label:
                        q.append((ni, nj))
    return matrix


def visualize(file_list,index):
    # Take for argument the name of both segmentations, in the right order ED then ES !
    for ind,file_name in enumerate(file_list) :
        seg_nii = nibabel.load(file_name)
        seg_data = np.asanyarray(seg_nii.dataobj, dtype=np.uint8)
        
        # N is the number of slices
        N = seg_data.shape[2]
        print(f"Number of slices : {N}")
        for slices in range(N):
            
            labels, counts = np.unique(seg_data[:,:,slices], return_counts=True)
            print(labels,counts)
            # Si on a detecter du mycordium : 
            if 2 in labels : 
                true_seg_copy = seg_data[:,:,slices].copy()
                print(f"Mycardium detected at slices : {slices}")
                # Compute the centroid
                X,Y = np.where(seg_data[:,:,slices]==2)
                X_centroid = int(np.mean(X))
                Y_centroid = int(np.mean(Y))
                
                # this remove the segmentation of the lV to be able to test my algo
                testing = True
                if testing :
                    x,y = np.where(seg_data[:,:,slices]==3)
                    for u,v in zip(x,y): 
                        seg_data[u,v,slices] = 0
               
                new_seg = flood_fill(seg_data[:,:,slices].copy(),(X_centroid,Y_centroid),3,2,0)
                error = np.sum(abs(new_seg-true_seg_copy))
                print(f" error made with segmentation technique : {error}")
                seg_data[:,:,slices] = new_seg
               

#a = visualize(["/Users/rplanchon/Documents/telecom/IMA/S2/IMA205/Challenge/CardiacPathoPrediction/Dataset/Train/001/001_ED_seg.nii","/Users/rplanchon/Documents/telecom/IMA/S2/IMA205/Challenge/CardiacPathoPrediction/Dataset/Train/001/001_ES_seg.nii"],"001")


# This function return the error made by segmenting with the techniques on 1 Training subject (ie 1 folder).
def evaluate_perf_seg_one(file_list,index):
    error = 0
    for ind,file_name in enumerate(file_list) :
        seg_nii = nibabel.load(file_name)
        seg_data = np.asanyarray(seg_nii.dataobj, dtype=np.uint8)
        # N is the number of slices
        N = seg_data.shape[2]
        #print(f"Number of slices : {N}")
        
        for slices in range(N):
           
            labels, counts = np.unique(seg_data[:,:,slices], return_counts=True)
            #print(f"Labels : {labels} counts : {counts} for slice {slices}")
            
            # Si on a detecter du mycordium : 
            if 2 in labels : 
                #print(f"Mycardium detected at slices : {slices}")

                # keep the true segmentation to evaluate performance.
                true_seg_copy = seg_data[:,:,slices].copy()
                
                # this remove the segmentation of the lV and replace it by bg to simulate test.
                x,y = np.where(seg_data[:,:,slices]==3)
                for u,v in zip(x,y): 
                    seg_data[u,v,slices] = 0
                        
                # Compute the centroid of the myocardium
                X,Y = np.where(seg_data[:,:,slices]==2)
                X_centroid = int(np.mean(X))
                Y_centroid = int(np.mean(Y))
                
                
                if seg_data[X_centroid,Y_centroid,slices] == 2:
                    print("centered point still inside..")
                    binary_seg = np.zeros(seg_data[:,:,slices].shape)
                    for ind,seg in np.ndenumerate(seg_data[:,:,slices]):
                        if seg == 2: 
                            binary_seg[ind] = 1
                    filled = binary_fill_holes(binary_seg.copy())
                    #print(filled)
                    A,B = np.where(filled==1)
                    C,D = np.where(binary_seg==1)

                    Z = np.zeros(seg_data[:,:,slices].shape)
                    for u,v in zip(A,B):
                        Z[u,v] = 1 
                    for t,y in zip(C,D) :
                        Z[t,y] = 0
                    F,G = np.where(Z==1)
                    
                    for f,g in zip(F,G):
                        seg_data[f,g,slices] = 3
                    
                    new_seg = seg_data[:,:,slices]           
                
                # fill the until reach myocardium
                else : 
                    new_seg = flood_fill(seg_data[:,:,slices].copy(),(X_centroid,Y_centroid),3,2,0)
                E =np.sum(abs(new_seg-true_seg_copy))
                if E> 0 : 
                    print(E) 
                    print(f"error at slice : {slices}, file name {file_name}")
                    pass
                error += E
                
    return error
    
def evaluate_perf_seg_total(root_train_folder_path): 
    TRAIN_DIR = root_train_folder_path
    total_error = 0

    for index in os.listdir(TRAIN_DIR): 
        folder_path = os.path.join(TRAIN_DIR,index)
        if os.path.isdir(folder_path):
            file_segED = str(index) + "_ED_SEG.nii"

            file_segES = str(index) + "_ES_SEG.nii"
            DIR_SEGED = os.path.join(folder_path,file_segED)
            DIR_SEGES = os.path.join(folder_path,file_segES) 
            total_error+= evaluate_perf_seg_one([DIR_SEGED,DIR_SEGES],index)
    
    return total_error

BASE_DIR = os.getcwd()
TRAIN_DIR = os.path.join(BASE_DIR,"Dataset/Train")

#print(evaluate_perf_seg_total(TRAIN_DIR)) 



def debug_one(file_name,slice):
    error = 0
    seg_nii = nibabel.load(file_name)
    seg_data = np.asanyarray(seg_nii.dataobj, dtype=np.uint8)
    # N is the number of slices
    N = seg_data.shape[2]
    print(f"Number of slices : {N}")
    
    labels, counts = np.unique(seg_data[:,:,slice], return_counts=True)
    print(f"Labels : {labels} counts : {counts} for slice {slice}")
    
    plt.figure()
    plt.imshow(seg_data[:,:,slice])
    plt.show()
    
    # Si on a detecter du mycordium : 
    if 2 in labels : 
        #print(f"Mycardium detected at slices : {slices}")

        # keep the true segmentation to evaluate performance.
        true_seg_copy = seg_data[:,:,slice].copy()
        
        # this remove the segmentation of the lV and replace it by bg to simulate test.
        x,y = np.where(seg_data[:,:,slice]==3)
        for u,v in zip(x,y): 
            seg_data[u,v,slice] = 0
                
        # Compute the centroid of the myocardium
        X,Y = np.where(seg_data[:,:,slice]==2)
        X_centroid = int(np.mean(X))
        Y_centroid = int(np.mean(Y))
        
        
        # fill the until reach myocardium
        if seg_data[X_centroid,Y_centroid,slice] == 2:
            print("centered point still inside..")
            binary_seg = np.zeros(seg_data[:,:,slice].shape)
            for ind,seg in np.ndenumerate(seg_data[:,:,slice]):
                if seg == 2: 
                    binary_seg[ind] = 1
            filled = binary_fill_holes(binary_seg.copy())
            #print(filled)
            A,B = np.where(filled==1)
            C,D = np.where(binary_seg==1)

            Z = np.zeros(seg_data[:,:,slice].shape)
            for u,v in zip(A,B):
                Z[u,v] = 1 
            for t,y in zip(C,D) :
                Z[t,y] = 0
            F,G = np.where(Z==1)
            
            for f,g in zip(F,G):
                seg_data[f,g,slice] = 3
            
            new_seg = seg_data[:,:,slice]           
        
        # fill the until reach myocardium
        else : 
            print("in center")
            new_seg = flood_fill(seg_data[:,:,slice].copy(),(X_centroid,Y_centroid),3,2,0)  
        E =np.sum(abs(new_seg-true_seg_copy))
        
        if E> 0 : 
            print(E) 
            print(f"error at slice : {slice}, file name {file_name}")
        error += E
        plt.figure()
        plt.imshow(new_seg)
        plt.show()
        
    return error

debug_one("/Users/rplanchon/Documents/telecom/IMA/S2/IMA205/Challenge/CardiacPathoPrediction/Dataset/Train/069/069_ES_SEG.nii",0)
import os
import math
import nibabel
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage import binary_fill_holes
from skimage import measure
from skimage.measure import label, regionprops
from matplotlib.patches import Circle
import nibabel as nib
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import shutil


def compute_dice_metric(true_seg,pred_seg,label = 3):
    # Create binary masks for the specific label
    true_seg_binary = (true_seg == label).astype(np.uint8)  # Mask for label in seg1
    pred_seg_binary = (pred_seg == label).astype(np.uint8)  # Mask for label in seg2

    # Compute the intersection and total sum
    intersection = np.sum(true_seg_binary * pred_seg_binary)  # Sum of element-wise multiplication
    total_sum = np.sum(true_seg_binary) + np.sum(pred_seg_binary)

    # Handle edge case to avoid dividing by zero
    if total_sum == 0:
        return 1.0 if np.all(true_seg_binary == pred_seg_binary) else 0.0

    # Calculate the Dice coefficient
    dice = 2.0 * intersection / total_sum
    return dice
  
def flood_fill(matrix, seed, new_label, boundary_label, max_dist):
    """
    Performs flood filling on `matrix` from seed until reaching cells with values in boundary_label
    or exceeding max_dist from seed. Uses 4-connected neighbors.
    """
    x, y = seed
    if (matrix[x, y] == boundary_label) or (matrix[x, y] == new_label):
        return matrix

    target_label = matrix[x, y]
    q = deque([seed])
    
    while q:
        i, j = q.popleft()
        if matrix[i, j] == target_label:
            matrix[i, j] = new_label
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < matrix.shape[0] and 0 <= nj < matrix.shape[1]:
                    if math.dist((ni, nj), seed) < max_dist:
                        if (matrix[ni, nj] not in boundary_label) and (matrix[ni, nj] == target_label):
                            q.append((ni, nj))
    return matrix

def my_seg(seg_data):
    """
    Performs the segmentation procedure on one segmentation volume (3D numpy array).
    Processing is done slice by slice. For slices containing myocardium (label 2):
      - Removes LV segmentation (label 3).
      - Computes the myocardium centroid.
      - Creates a binary mask of the myocardium and fills its holes.
      - If the filled mask equals the binary mask (i.e. no extra pixels are added by filling),
        performs a backup procedure using contours and flood_fill (assigning label 3).
      - Otherwise, assigns label 3 to pixels in the filled mask not in the binary myocardium.
    
    Parameters:
      seg_data: 3D numpy array containing the segmentation data.
    
    Returns:
      new_seg_data: 3D numpy array with the updated segmentation (same type and shape as seg_data).
    """
    new_seg_data = seg_data.copy()
    num_slices = seg_data.shape[2]
    
    for s in range(num_slices):
        slice_data = seg_data[:, :, s].copy()
        if 2 not in np.unique(slice_data):
            continue  # Skip slices without myocardium
        
        # Remove LV segmentation (label 3)
        slice_data[slice_data == 3] = 0
        
        # Compute the centroid of myocardium (label 2)
        indices = np.where(slice_data == 2)
        if len(indices[0]) == 0:
            continue
        X_centroid = int(np.mean(indices[0]))
        Y_centroid = int(np.mean(indices[1]))
        
        # Create a binary mask for myocardium (label 2) and fill its holes
        binary_seg = (slice_data == 2).astype(np.uint8)
        filled = binary_fill_holes(binary_seg.copy())
        
        # If filling doesn't add extra pixels, use the backup procedure with contour extraction.
        if np.sum(filled) == np.sum(binary_seg):
          
            labeled_binary_seg = label(binary_seg)
            props = regionprops(labeled_binary_seg)
            if not props:
                continue
            region_obj = max(props, key=lambda r: r.area)
            centroid = region_obj.centroid

            contours = measure.find_contours(binary_seg, level=0.5)
            def contour_mean_distance(contour, center):
                return np.mean(np.sqrt((contour[:, 0] - center[0])**2 +
                                       (contour[:, 1] - center[1])**2))
            distances = [contour_mean_distance(contour, centroid) for contour in contours]
            inner_contour = contours[np.argmin(distances)]
            inner_radii = np.sqrt((inner_contour[:, 0] - centroid[0])**2 +
                                  (inner_contour[:, 1] - centroid[1])**2)
            inner_radius_est = np.mean(inner_radii)
            #print("Estimated inner radius:", inner_radius_est)
            
            new_slice = flood_fill(slice_data.copy(),
                                   (X_centroid, Y_centroid),
                                   3,
                                   [2],
                                   inner_radius_est)
        else:
            # Otherwise, update pixels where the filled mask is 1 but the binary mask is 0.
            new_slice = slice_data.copy()
            mask = (filled == 1) & (binary_seg == 0)
            new_slice[mask] = 3
        
        new_seg_data[:, :, s] = new_slice
        
    
    return new_seg_data

def evaluate_my_seg_one(file_list, index):
    """
    For one subject (given a list of segmentation file paths), evaluates the segmentation procedure.
    For each file:
      - Loads the segmentation volume.
      - Applies my_seg to generate the new segmentation.
      - For slices containing myocardium, computes the pixel-wise error between the original and new segmentation.
    
    Parameters:
      file_list: List of file paths for the subject's segmentation volumes.
      index: Subject identifier.
    
    Returns:
      error: Total error (sum over all slices and files) computed as the number of differing pixels.
    """
    error = []
    worst_error = []
    # N the total number of slices
    N = 0
    for file_name in file_list:
        seg_nii = nibabel.load(file_name)
        seg_data = np.asanyarray(seg_nii.dataobj, dtype=np.uint8)
        
        # Apply my_seg to the segmentation volume.
        new_seg = my_seg(seg_data)
        
        # Evaluate error slice by slice (only for slices containing myocardium).
        N += seg_data.shape[2]
        for s in range(seg_data.shape[2]):
            if 2 not in np.unique(seg_data[:, :, s]):
                continue
            true_seg = seg_data[:, :, s].copy()
            new_seg_slice = new_seg[:, :, s]
            
            # Here compute the dice metric : 
            E = compute_dice_metric(true_seg,new_seg_slice)
            # E = np.sum(new_seg_slice != true_seg)
            if E !=1 : 
              worst_error.append(E)
            error.append(E)
    return np.sum(error)/len(error),worst_error,N

def evaluate_my_seg_total(root_train_folder_path):
    """
    Evaluates the segmentation performance over all subjects in the dataset.
    Iterates over each subject folder in the given root directory (e.g., Train folder),
    expecting each subject to have segmentation files such as <index>_ED_SEG.nii and <index>_ES_SEG.nii.
    
    Parameters:
      root_train_folder_path: Path to the root folder containing subject subfolders.
    
    Returns:
      total_error: The sum of errors over all subjects.
    """
    total_error = 0
    total_number_slices = 0
    error_non_surrounded_case = []
    for index in os.listdir(root_train_folder_path):
        folder_path = os.path.join(root_train_folder_path, index)
        if os.path.isdir(folder_path):
            file_segED = f"{index}_ED_SEG.nii"
            file_segES = f"{index}_ES_SEG.nii"
            DIR_SEGED = os.path.join(folder_path, file_segED)
            DIR_SEGES = os.path.join(folder_path, file_segES)
            file_list = [DIR_SEGED, DIR_SEGES]
            result =evaluate_my_seg_one(file_list, index) 
            total_error += result[0]
            total_number_slices += result[2]
            error_non_surrounded_case.extend(result[1])
            
    #print(np.mean(error_non_surrounded_case),np.min(error_non_surrounded_case),np.max(error_non_surrounded_case))
    #print(f"total number of slices", total_number_slices)
    return total_error


def debug_one(seg_file_name, slice_index, save_fig=False, display=True, save_dir="./debug_outputs"):
    """
    Debugs and visualizes the segmentation for a single file and slice.
    Optionally saves figures
    
    Parameters:
      seg_file_name: Path to the segmentation file (*.nii or *.nii.gz)
      slice_index: Slice index to visualize
      save_fig: If True, saves figures instead of or in addition to displaying them
      display: If True, shows the figures on screen
      save_dir: Directory where to save images if save_fig is True
      
    Returns:
      error: Number of differing pixels between true and predicted segmentations.
    """
    # Load segmentation
    seg_nii = nib.load(seg_file_name)
    seg_data = np.asanyarray(seg_nii.dataobj, dtype=np.uint8)
    num_slices = seg_data.shape[2]
    #print(f"Number of slices: {num_slices}")

    # Derive MRI filename
    base_path, seg_filename = os.path.split(seg_file_name)
    mri_filename = seg_filename.replace('_SEG', '')
    mri_file_path = os.path.join(base_path, mri_filename)
    
    if not os.path.exists(mri_file_path):
        raise FileNotFoundError(f"Original MRI file not found: {mri_file_path}")

    # Load MRI image
    mri_nii = nib.load(mri_file_path)
    mri_data = np.asanyarray(mri_nii.dataobj, dtype=np.float32)
    background_slice = mri_data[:, :, slice_index]
    background_slice = background_slice / np.max(background_slice)  # Normalize for display

    # Extract true and predicted segmentations
    true_seg = seg_data[:, :, slice_index]
    new_seg = my_seg(seg_data)[:, :, slice_index]

    # Compute error
    E = np.sum(new_seg != true_seg)
    if E > 0:
        print(f"Error at slice {slice_index}, file {seg_file_name}: {E} differing pixels")
    
    # Define labels and colors
    label_map = {
        0: "Background",
        1: "LV",
        2: "RV",
        3: "Myocardium"
    }
    color_map = {
        0: "black",
        1: "red",
        2: "blue",
        3: "green"
    }

    def save_or_show_segmentation(background, seg_slice, title, save_name=None):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(background, cmap='gray', interpolation='none')

        # Build colored segmentation
        colored_seg = np.zeros(seg_slice.shape + (3,), dtype=np.float32)
        for label_id, color_name in color_map.items():
            mask = (seg_slice == label_id)
            color_rgb = np.array(mcolors.to_rgb(color_name))
            colored_seg[mask] = color_rgb

        ax.imshow(colored_seg, alpha=0.5, interpolation='none')
        ax.axis("off")
        plt.tight_layout()

        if save_fig and save_name:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            path = os.path.join(save_dir, save_name)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {path}")

        if display:
            plt.title(title)
            plt.show()
        else:
            plt.close()

    def save_legend(save_name="legend.png"):
        fig_legend, ax_legend = plt.subplots(figsize=(3, 2))

        handles = [
            mpatches.Patch(color=color_map[label_id], label=label_map[label_id])
            for label_id in color_map
        ]
        ax_legend.legend(handles=handles, loc='center')
        ax_legend.axis('off')
        plt.tight_layout()

        if save_fig:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            path = os.path.join(save_dir, save_name)
            plt.savefig(path, dpi=300, bbox_inches='tight',transparent=True)
            print(f"Saved legend to {path}")

        if display:
            plt.show()
        else:
            plt.close()

    # Display or save both segmentations
    save_or_show_segmentation(background_slice, true_seg, "True Segmentation Overlay", save_name="true_seg.png")
    save_or_show_segmentation(background_slice, new_seg, "Predicted Segmentation Overlay", save_name="predicted_seg.png")
    save_legend()

    return E

def segment_and_save_test_set(input_test_folder, output_folder):
    # Segment the test entire test folder. 
    # Add the left ventricule labels
    # Create a new test set in ouput folder location where the segmentation is completed.
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for index in os.listdir(input_test_folder):
        subject_path = os.path.join(input_test_folder, index)
        if os.path.isdir(subject_path):
            output_subject_path = os.path.join(output_folder, index)
            os.makedirs(output_subject_path, exist_ok=True)

            for file in os.listdir(subject_path):
                input_path = os.path.join(subject_path, file)
                output_path = os.path.join(output_subject_path, file)

                if file.endswith('_ED_seg.nii') or file.endswith('_ES_seg.nii'):
                    # Process segmentation files
                    seg_nii = nib.load(input_path)
                    seg_data = np.asanyarray(seg_nii.dataobj, dtype=np.uint8)
                    new_seg = my_seg(seg_data)

                    new_nii = nib.Nifti1Image(new_seg, affine=seg_nii.affine, header=seg_nii.header)
                    nib.save(new_nii, output_path)
                    print(f"Segmented and saved: {output_path}")
                else:
                    # Copy non-segmentation files
                    shutil.copy2(input_path, output_path)
                    print(f"Copied original image: {output_path}")

# # To evaluate on all subjects in the Train folder:
# BASE_DIR = os.getcwd()
# TRAIN_DIR = os.path.join(BASE_DIR, "Dataset/Train")
# TEST_DIR = os.path.join(BASE_DIR,"Dataset/Test")
# total_error = evaluate_my_seg_total(TRAIN_DIR)
# print("Mean dice over the trainning:", total_error)

# To compute the new Testing folder 

# SEGMENTED_TEST_DIR = os.path.join(BASE_DIR,"Dataset/SegTest") 
# segment_and_save_test_set(TEST_DIR,SEGMENTED_TEST_DIR)


# closed example

# debug_error = debug_one("/Users/rplanchon/Documents/telecom/IMA/S2/IMA205/Challenge/CardiacPathoPrediction/Dataset/Train/020/020_ES_SEG.nii",5,save_fig=False)
# print("Debug slice error:", debug_error)


# Non closed example

# debug_error = debug_one("/Users/rplanchon/Documents/telecom/IMA/S2/IMA205/Challenge/CardiacPathoPrediction/Dataset/Train/029/029_ED_SEG.nii",0,save_fig=True)
# print("Debug slice error:", debug_error)

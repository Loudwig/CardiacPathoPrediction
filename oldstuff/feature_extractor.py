# extract_shape.py
import os, numpy as np, pandas as pd, SimpleITK as sitk
from radiomics import featureextractor

# ─────── 1. Shape-only extractor (no YAML, no params dict) ───────
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.settings['additionalInfo'] = False
extractor.disableAllImageTypes()
extractor.enableImageTypeByName('Original')
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('shape')


print("Image types :", extractor.enabledImagetypes)          # {'Original': {}}
print("Feature cls :", extractor.enabledFeatures.keys())     # dict_keys(['shape'])
print("#shape feats :", len(extractor.enabledFeatures['shape']))  # 14

# ─────── 2. I/O folders ───────
BASE       = os.getcwd()
TRAIN_DIR  = os.path.join(BASE, "Dataset", "Train")
TEST_DIR   = os.path.join(BASE, "Dataset", "SegTest")
OUT_DIR    = os.path.join(BASE, "data")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────── 3. per-subject extraction ───────
def do_subject(folder, sid):
    rec = {"Id": sid}
    for ph in ["ED", "ES"]:
        img = sitk.ReadImage(os.path.join(folder, f"{sid}_{ph}.nii"))
        msk = sitk.ReadImage(os.path.join(folder, f"{sid}_{ph}_seg.nii"))
        for lbl, name in [(1,"RV"),(3,"LV"),(2,"MY")]:
            try:
                feats = extractor.execute(img, msk, label=lbl)
                for k,v in feats.items():
                    rec[f"{ph}_{name}_{k.replace('original_','')}"] = v
            except ValueError:
                # label not present → fill NaN for all 14 shape metrics
                for k in extractor.enabledFeatures['shape']:
                    rec[f"{ph}_{name}_{k}"] = np.nan
    return rec

def build_csv(src_dir, out_csv):
    rows = [do_subject(os.path.join(src_dir,s), s)
            for s in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir,s))]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"{out_csv}: {df.shape[0]} rows × {df.shape[1]-1} shape cols (expect 84)")

build_csv(TRAIN_DIR, os.path.join(OUT_DIR,"Training_shape.csv"))
build_csv(TEST_DIR,  os.path.join(OUT_DIR,"Testing_shape.csv"))

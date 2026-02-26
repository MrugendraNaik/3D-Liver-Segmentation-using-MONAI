import os
import nibabel as nib
import numpy as np

# ==== EDIT THESE ====
img_path = r"D:/LIVERLAST/ATRAIN/imagesTr/liver_77.nii.gz"
lab_path = r"D:/LIVERLAST/ATRAIN/labelsTr/liver_77.nii.gz"

OUT_IMAGES = r"D:/LIVERLAST/ATRAIN/images"
OUT_LABELS = r"D:/LIVERLAST/ATRAIN/labels"

CHUNK_SIZE = 64
new_name = "liver_XX_tail_0"   # output file prefix
# ====================

os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)

img_nii = nib.load(img_path)
lab_nii = nib.load(lab_path)

img = img_nii.get_fdata()
lab = lab_nii.get_fdata().astype(np.uint8)

if img.shape != lab.shape:
    raise ValueError(f"Shape mismatch: img {img.shape} vs lab {lab.shape}")

H, W, D = img.shape
if D < CHUNK_SIZE:
    raise ValueError(f"Volume depth {D} is less than CHUNK_SIZE {CHUNK_SIZE}")

start = D - CHUNK_SIZE
end = D

img_chunk = img[:, :, start:end]
lab_chunk = lab[:, :, start:end]

out_img_path = os.path.join(OUT_IMAGES, new_name + ".nii.gz")
out_lab_path = os.path.join(OUT_LABELS, new_name + ".nii.gz")

nib.save(nib.Nifti1Image(img_chunk, img_nii.affine), out_img_path)
nib.save(nib.Nifti1Image(lab_chunk, lab_nii.affine), out_lab_path)

print(f"Saved tail chunk slices [{start}:{end})")
print("Image:", out_img_path)
print("Label:", out_lab_path)
print("Unique labels in chunk:", np.unique(lab_chunk))

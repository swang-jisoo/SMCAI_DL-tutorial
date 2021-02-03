# Import necessary libraries
import os


# parameters
subtypes = ['LAA', 'CE', 'SVO']
root_dir = 'D:/Dropbox/ESUS_ML/SSAI_STROKE'


def get_matched_fpath(mask_dir, img_dir):
    """ Return a list of path of input images containing 'DWI' in the 'input' folder under the given directory. """
    # Get file names in the folder
    (_, _, f_mask) = next(os.walk(mask_dir))
    (_, _, f_img) = next(os.walk(img_dir))
    f_mask.sort()
    f_img.sort()
    dwi_mask, dwi_nomask, adcdwi_mask, adcdwi_nomask, etc = [], [], [], [], []
    for i in f_img:
        if 'DWI' in i:
            f_dwi = os.path.splitext(i)[0]
            f_adc = f_dwi[:f_dwi.index('DWI')] + 'ADC' + f_dwi[f_dwi.index('DWI') + 3:]
            if (f_adc + '.dcm' not in f_img) and (f_dwi + '.png' in f_mask):
                # fname_dwi has file names containing 'DWI', which have corresponding mask files
                dwi_mask.append(i)
            elif (f_adc + '.dcm' not in f_img) and (f_dwi + '.png' not in f_mask):
                dwi_nomask.append(i)
            elif (f_adc + '.dcm' in f_img) and (f_dwi + '.png' in f_mask):
                # fname_dwi has file names containing 'DWI', which have corresponding 'ADC' and mask files
                adcdwi_mask.append(i)
            elif (f_adc + '.dcm' in f_img) and (f_dwi + '.png' not in f_mask):
                adcdwi_nomask.append(i)
        else:
            etc.append(i)
    return f_img, dwi_mask, dwi_nomask, adcdwi_mask, adcdwi_nomask, etc  # {folder_dir: fname_dwi}


mask_dir = os.path.join(root_dir, 'GT')

subtype_idx = 0
img_dir = os.path.join(root_dir, subtypes[subtype_idx])
f_img, dwi_mask, dwi_nomask, adcdwi_mask, adcdwi_nomask, etc = get_matched_fpath(mask_dir, img_dir)
print(len(f_img), len(dwi_mask), len(dwi_nomask), len(adcdwi_mask), len(adcdwi_nomask), len(etc))
print(len(dwi_mask) + len(dwi_nomask) + (len(adcdwi_mask) + len(adcdwi_nomask))*2)
filePath = './LAA_ADCDWI_wo_mask.txt'
with open(filePath, 'w+') as lf:
    lf.write('\n'.join(adcdwi_nomask))

subtype_idx = 1
img_dir = os.path.join(root_dir, subtypes[subtype_idx])
f_img, dwi_mask, dwi_nomask, adcdwi_mask, adcdwi_nomask, etc = get_matched_fpath(mask_dir, img_dir)
print(len(f_img), len(dwi_mask), len(dwi_nomask), len(adcdwi_mask), len(adcdwi_nomask), len(etc))
print(len(dwi_mask) + len(dwi_nomask) + (len(adcdwi_mask) + len(adcdwi_nomask))*2)
filePath = './CE_ADCDWI_wo_mask.txt'
with open(filePath, 'w+') as lf:
    lf.write('\n'.join(adcdwi_nomask))

for e in etc:
    f_dwi = e[:e.index('ADC')] + 'DWI' + e[e.index('ADC') + 3:]
    if (f_dwi not in adcdwi_mask) and (f_dwi not in adcdwi_nomask):
        print(e)

subtype_idx = 2
img_dir = os.path.join(root_dir, subtypes[subtype_idx])
f_img, dwi_mask, dwi_nomask, adcdwi_mask, adcdwi_nomask, etc = get_matched_fpath(mask_dir, img_dir)
print(len(f_img), len(dwi_mask), len(dwi_nomask), len(adcdwi_mask), len(adcdwi_nomask), len(etc))
print(len(dwi_mask) + len(dwi_nomask) + (len(adcdwi_mask) + len(adcdwi_nomask))*2)
filePath = './SVO_ADCDWI_wo_mask.txt'
with open(filePath, 'w+') as lf:
    lf.write('\n'.join(adcdwi_nomask))

for e in etc:
    f_dwi = e[:e.index('ADC')] + 'DWI' + e[e.index('ADC') + 3:]
    if (f_dwi not in adcdwi_mask) and (f_dwi not in adcdwi_nomask):
        print(e)
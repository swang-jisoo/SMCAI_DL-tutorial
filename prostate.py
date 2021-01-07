#####
# Image semantic segmentation

# Dataset: harvard dataverse prostate
# ref. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP

# Model: U-net

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# Load Harvard dataverse prostate dataset
# 1. Download all 15 tar.gz files from the link below:
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP

# 2. Understand the directory structure
#   + dataverse_files
#       + Gleason_masks_test_pathologist1.tar.gz: test masks 1; png files
#       + Gleason_masks_test_pathologist2.tar.gz: test masks 2; png files
#       + ZT80_38_A~C.tar.gz: as all together, test images; jpg files
#       + Gleason_masks_train.tar.gz: train masks; png files
#       + rest of ZTxx_x_x.tar.gz: as all together, train images; jpg files

# Some of the images does not have the corresponding masks, so check th


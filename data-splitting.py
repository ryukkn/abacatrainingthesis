import splitfolders

import os
import cv2
import numpy as np
from random import sample
import shutil

split = 0.8

# remove existing split
# try:
#     shutil.rmtree("./dataset")
# except:
#     print("No exisiting split.")
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.

print("Splitting Data.....")
splitfolders.ratio("../../../../nodeserver/data/grades", output="dataset",
    seed=1337, ratio=(split, (1-split)/2, (1-split)/2),group_prefix=None, move=False) # default values


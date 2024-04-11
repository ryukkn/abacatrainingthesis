import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("../../../../nodeserver/data/grades", output="dataset",
    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # default values

# balanced
# splitfolders.fixed("input_folder", output="output",
#     seed=1337, fixed=(100, 100), oversample=False, group_prefix=None, move=False) # default values
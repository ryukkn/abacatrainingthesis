import os

def balance_directory(root_dir):
    # Get list of subdirectories
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    # Find the subdirectory with the maximum number of files
    max_files = 999999999999999999999
    max_dir = ''
    perclass ={}
    for subdir in subdirs:
        num_files = len(os.listdir(os.path.join(root_dir, subdir)))
        perclass[subdir] = num_files
        if num_files < max_files:
            max_files = num_files
            max_dir = subdir

    # Remove excess files from the directory with the maximum number of files
    for subdir in subdirs:
        if subdir == max_dir:
            continue
        source_dir = os.path.join(root_dir, subdir)
        files_to_remove = os.listdir(source_dir)[max_files:]
        for file_to_remove in files_to_remove:
            os.remove(os.path.join(source_dir, file_to_remove))
            continue

    print("Directory balanced successfully!")

# Example usage:
balance_directory("./dataset/train")
balance_directory("./dataset/val")
# balance_directory("./dataset/augmented")
balance_directory("./dataset/test")
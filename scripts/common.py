import os
from glob import glob

from tqdm import tqdm
from loguru import logger


def cleanup_webdataset_directory(dir_name):
    """Cleans up a directory of json and jpeg files, deleting any files that don't have a matching pair."""
    json_files = set()
    jpeg_files = set()
    n_json_deleted = 0
    n_jpeg_deleted = 0
    for f in tqdm(glob(dir_name + "/*"), desc=f"files in {dir_name}"):
        # the first name will be the directory name, we want to skip that
        if f == dir_name:
            continue

        if f.endswith(".json"):
            json_files.add(f)
        elif f.endswith(".jpg"):
            jpeg_files.add(f)
        else:
            raise ValueError("Unexpected file type: " + f)

    for f in json_files:
        jpeg_filename = f[:-5] + ".jpg"
        if jpeg_filename not in jpeg_files:
            os.remove(f)
            n_json_deleted += 1

    for f in jpeg_files:
        json_filename = f[:-4] + ".json"
        if json_filename not in json_files:
            os.remove(f)
            n_jpeg_deleted += 1
    
    n_files_processed = len(json_files)

    if n_json_deleted + n_jpeg_deleted == 0:
        return n_json_deleted, n_jpeg_deleted, n_files_processed

    # Verify that the number of json files and jpeg files in the directory is the same now
    json_files = glob(f"{dir_name}/*.json")
    jpeg_files = glob(f"{dir_name}/*.jpg")
    if len(json_files) != len(jpeg_files):
        logger.info(f"json_files: {json_files}")
        logger.info(f"jpeg_files: {jpeg_files}")
        raise RuntimeError("Number of json files and jpeg files is not the same")

    return n_json_deleted, n_jpeg_deleted, n_files_processed

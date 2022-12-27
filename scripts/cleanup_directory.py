import argparse
from loguru import logger
from ldm.texty_caps_utils import cleanup_webdataset_directory



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # use first positional argument as input directory
    parser.add_argument("input_dir", help="path to directory with images", type=str)
    args = parser.parse_args()

    n_json_deleted, n_jpeg_deleted, n_files_processed = cleanup_webdataset_directory(args.input_dir)

    logger.info(f"Deleted {n_json_deleted} json files and {n_jpeg_deleted} jpeg files from {args.input_dir}")
    logger.info(f"Processed {n_files_processed} files in {args.input_dir}")

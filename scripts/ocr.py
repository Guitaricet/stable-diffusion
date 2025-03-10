"""Perform OCR on images, check if ocr'ed text is close to caption, if no delete image if yes move to the dataset folder

export CUDA_VISIBLE_DEVICES=0  # the script will use all available GPUs in an inefficient way, so we limit it to one
python scripts/ocr.py \
    --input_dir "/home/vlialin/data/texty-caps" \
    --output_dir "/home/vlialin/data/texty-caps-ocred" \
    --start_shard 166 \
    --process_every 2 \
    --num_shards 50

export CUDA_VISIBLE_DEVICES=1  # the script will use all available GPUs in an inefficient way, so we limit it to one
python scripts/ocr.py \
    --input_dir "/home/vlialin/data/texty-caps" \
    --output_dir "/home/vlialin/data/texty-caps-ocred" \
    --start_shard 167 \
    --process_every 2 \
    --num_shards 50

"""
import argparse
import time
import os
import json
import subprocess
from glob import glob

import numpy as np

import evaluate
import easyocr
from tqdm.auto import tqdm
from loguru import logger

from ldm.texty_caps_utils import cleanup_webdataset_directory


reader = easyocr.Reader(["en"], detect_network='craft', gpu=True)
chrf = evaluate.load("chrf")


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="path to directory with images", required=True)
    parser.add_argument("--output_dir", help="path to directory where to save images", required=True)

    parser.add_argument("--start_shard", required=True, type=int)
    parser.add_argument("--process_every", default=1, type=int)
    parser.add_argument("--num_shards", required=True, type=int)

    parser.add_argument("--gcp_bucket_name", default=None, type=str)

    return parser.parse_args()


def main(args):
    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    script_start_time = time.time()

    total_images_with_ocr = 0
    processed_shard_names = []
    for shard_id in range(args.start_shard, args.start_shard + args.num_shards, args.process_every):
        shard_name = f"shard_{shard_id:06d}"

        shard_start_time = time.time()
        input_dir = os.path.join(args.input_dir, shard_name)
        output_dir = os.path.join(args.output_dir, shard_name)

        logger.info(f"Processing shard {shard_name} in directory {input_dir}")

        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory {input_dir} does not exist")

        shard_images_with_ocr = process_shard(input_dir, output_dir)

        # delete any files that don't have a matching pair
        # this might happen if you ran script over the same directory twice
        cleanup_webdataset_directory(output_dir)

        if args.gcp_bucket_name is not None:
            # tar and upload to GCP bucket asychronously
            output_dir_base_name = os.path.basename(output_dir)

            # e.g., tar --sort=name -cf output_dir.tar -C output_dir output_dir_base_name
            tar_command = f"tar --sort=name -cf {output_dir}.tar -C {output_dir} {output_dir_base_name}"
            upload_command = f"gsutil cp {output_dir}.tar gs://{args.gcp_bucket_name}/{shard_name}.tar"
            delete_directory_command = f"rm -rf {output_dir}"
            delete_tar_command = f"rm -rf {output_dir}.tar"

            # do not run the next command if the previous one failed
            # echo between all commands
            full_command = f"""
            {tar_command} && \
                echo "Tarred directory {output_dir}" && \
                {upload_command} && \
                echo "Uploaded tar to GCP bucket" && \
                {delete_directory_command} && \
                echo "Deleted directory {output_dir}" && \
                {delete_tar_command} && \
                echo "Deleted tar {output_dir}.tar" && \
                echo "Finished tar-upload process for shard {shard_name}"
            """
            output = subprocess.run(full_command, shell=True)
            if output[0].split()[-1] != "Successfully":
                raise RuntimeError("Upload failed: " + output)

        total_images_with_ocr += shard_images_with_ocr
        processed_shard_names.append(shard_name)

        hours = (time.time() - shard_start_time) / 3600
        logger.info(f"Processed shard {shard_name} in {hours:.2f} hours")
        logger.info(f"Found {shard_images_with_ocr} images with text in shard {shard_name}")
    
    hours = (time.time() - script_start_time) / 3600
    logger.info(f"Processed all shards in {hours:.2f} hours")
    logger.info(f"Found {total_images_with_ocr} images with text in total")
    logger.info(f"Processed shards: {processed_shard_names}")


def format_result(ocr_result):
    res = []
    for line in ocr_result:
        bbox = np.array(line[0]).tolist()  # converts np.int64 to python int to make it json-serializable
        text = line[1]
        confidence = float(line[2])
        res.append({"bbox": bbox, "text": text, "confidence": confidence})
    return res


def process_shard(shard_dir, shard_output_dir):
    os.makedirs(shard_output_dir, exist_ok=True)

    _shard_start_time = time.time()
    no_meta_count = 0
    n_images_with_ocr = 0
    total_images_processed = 0
    all_subfolders = sorted(glob(os.path.join(shard_dir, "*")))

    pbar = tqdm(all_subfolders)
    for subfolder in pbar:
        # check if subfolder is actualy a folder
        if not os.path.isdir(subfolder):
            continue
        if total_images_processed > 0:
            # update progress bar description
            images_per_second = total_images_processed / (time.time() - _shard_start_time)
            pbar.set_description(f"Processed {total_images_processed} images, {images_per_second:.2f} images/s")

        subfolder_images_with_ocr = 0
        for image_path in glob(f"{subfolder}/*.jpg"):
            image_save_path = os.path.join(shard_output_dir, os.path.basename(image_path))
            if os.path.exists(image_save_path):
                continue

            image_name = os.path.basename(image_path).split(".")[0]

            image_meta_path = os.path.join(subfolder, f"{image_name}.json")
            txt_meta_path = os.path.join(subfolder, f"{image_name}.txt")
            if not os.path.exists(image_meta_path):
                logger.warning(f"\t\tNo meta file for image {image_path}")
                no_meta_count += 1
                continue

            with open(image_meta_path, "r") as f:
                image_meta = json.load(f)

            caption = image_meta["caption"]

            if caption is None or caption == "": # happens a few times in the dataset
                os.remove(image_path)
                os.remove(image_meta_path)
                os.remove(txt_meta_path)
                continue

            caption = caption.lower()

            ocr_output = reader.readtext(
                image_path,
                batch_size=64,  # maximizes speed in our experiments (RTX 3090)
            )

            all_image_text = ""
            for ocr_item in ocr_output:
                ocr_confidence = ocr_item[2]
                if ocr_confidence < 0.5:
                    continue

                ocr_text = ocr_item[1].lower()
                all_image_text += ocr_text + " "

            # compute how many char n-grams (up to 4) are in common
            # between ocr'ed text and caption
            # similarity is a number between 0 and 100, but usually it is about 10-20 for high-quality image-caption pairs
            similarity = chrf.compute(predictions=[all_image_text], references=[caption], beta=0, char_order=4, word_order=0)["score"]
            if caption is not None and similarity > 0.9:  # very weak filtering, just to remove absolute trash
                # move image to the dataset folder
                n_images_with_ocr += 1
                subfolder_images_with_ocr += 1

                # convert ocr_output from numpy to python types
                image_meta["ocr"] = format_result(ocr_output)  # first versions of the dataset (before Oct 8 2022) don't have this field
                image_meta["ocap"] = float(similarity)
                _new_meta_path = os.path.join(shard_output_dir, os.path.basename(image_meta_path))
                with open(_new_meta_path, "w") as f:
                    json.dump(image_meta, f)

                os.rename(image_path, os.path.join(shard_output_dir, os.path.basename(image_path)))

            if os.path.exists(image_path): os.remove(image_path)
            os.remove(image_meta_path)
            os.remove(txt_meta_path)

            total_images_processed += 1
    
    shards = glob(os.path.join(shard_output_dir, "*"))
    total_images = 0
    for shard in shards:
        if os.path.isdir(shard):
            total_images += len(glob(os.path.join(shard, "*.jpg")))
        else:
            total_images += 1

    minutes = (time.time() - _shard_start_time) / 60
    logger.info(f"Processed {len(all_subfolders)} subfolders in {minutes:.2f} minutes")
    logger.info(f"Found {n_images_with_ocr} images with text in total")
    logger.warning(f"Skipped {no_meta_count} images without .json file")
    logger.info(f"Total images produced in shard: {total_images}")
    return n_images_with_ocr


if __name__ == "__main__":
    args = make_args()
    main(args)
    logger.info("Script finished successfully")

"""Perform OCR on images, check if ocr'ed text is close to caption, if no delete image if yes move to the dataset folder

python scripts/ocr.py \
    --input_dir "/home/vladislavlialin/data/text-laion-20M-images" \
    --output_dir "/home/vladislavlialin/data/text-laion-20M-images-with-text" \
    --start_shard 100 \
    --num_shards 100
"""
import argparse
import time
import os
import json
from glob import glob

import evaluate
import easyocr
from tqdm.auto import tqdm

reader = easyocr.Reader(["en"], detect_network='craft', gpu=True)
chrf = evaluate.load("chrf")


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="path to directory with images", required=True)
    parser.add_argument("--output_dir", help="path to directory where to save images", required=True)

    parser.add_argument("--start_shard", default=None, type=int)
    parser.add_argument("--num_shards", default=None, type=int)

    return parser.parse_args()


def main(args):
    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    script_start_time = time.time()

    if args.start_shard is None:
        process_shard(args.input_dir, args.output_dir)

    total_images_with_ocr = 0
    if args.start_shard is not None:
        for shard_id in range(args.start_shard, args.start_shard + args.num_shards):
            shard_start_time = time.time()
            input_dir = os.path.join(args.input_dir, f"shard_{shard_id:06d}")
            output_dir = os.path.join(args.output_dir, f"shard_{shard_id:06d}")

            print(f"Processing shard {shard_id} in directory {input_dir}")
            shard_images_with_ocr = process_shard(input_dir, output_dir)
            total_images_with_ocr += shard_images_with_ocr

            hours = (time.time() - shard_start_time) / 3600
            print(f"Processed shard {shard_id} in {hours:.2f} hours")
            print(f"Found {shard_images_with_ocr} images with text in shard {shard_id}")
    
    hours = (time.time() - script_start_time) / 3600
    print(f"Processed all shards in {hours:.2f} hours")
    print(f"Found {total_images_with_ocr} images with text in total")


def process_shard(shard_dir, shard_output_dir):
    os.makedirs(shard_output_dir, exist_ok=True)

    n_images_with_ocr = 0
    all_subfolders = sorted(glob(os.path.join(shard_dir, "*")))
    for subfolder in tqdm(all_subfolders, desc="subfolders"):
        # check if subfolder is actualy a folder
        if not os.path.isdir(subfolder):
            continue

        print(f"Processing {subfolder}")
        subfolder_start_time = time.time()
        subfolder_images_with_ocr = 0
        num_subfolder_images = len(glob(os.path.join(subfolder, "*.jpg")))
        for image_path in tqdm(glob(f"{subfolder}/*.jpg"), desc="images in subfolder"):
            image_save_path = os.path.join(shard_output_dir, os.path.basename(image_path))
            if os.path.exists(image_save_path):
                continue

            image_name = os.path.basename(image_path).split(".")[0]

            image_meta_path = os.path.join(subfolder, f"{image_name}.json")
            with open(image_meta_path, "r") as f:
                image_meta = json.load(f)

            ocr_output = reader.readtext(
                image_path,
                batch_size=64,  # maximizes speed in our experiments (3090)
            )

            for ocr_item in ocr_output:
                ocr_confidence = ocr_item[2]
                if ocr_confidence < 0.7:
                    continue

                ocr_text = ocr_item[1].lower()
                item_text = image_meta["caption"]

                if item_text is None: # happens a few times in a dataset
                    break

                item_text = item_text.lower()

                # compute how many char n-grams (up to 4) are in common
                # between ocr'ed text and caption

                similarity = chrf.compute(predictions=[ocr_text], references=[item_text])["score"]
                if similarity > 0.9:
                    # move image to the dataset folder
                    n_images_with_ocr += 1
                    subfolder_images_with_ocr += 1
                    os.rename(image_path, os.path.join(shard_output_dir, os.path.basename(image_path)))

                    image_meta["ocr"] = ocr_output  # first versions of the dataset (before Oct 8 2022) don't have this field
                    with os.path.join(shard_output_dir, os.path.basename(image_meta_path)) as f:
                        json.dump(image_meta, f)

                    break
            else:
                # delete image
                os.remove(image_path)
                os.remove(image_meta_path)

        hours = (time.time() - subfolder_start_time) / 3600
        print(f"Processed {subfolder} in {hours:.2f} hours")
        print(f"Found {subfolder_images_with_ocr} images with text in {subfolder} out of {num_subfolder_images} images")
        print(f"In total found {n_images_with_ocr} images with text")

    print(f"Processed {len(all_subfolders)} subfolders")
    print(f"Found {n_images_with_ocr} images with text in total")
    return n_images_with_ocr


if __name__ == "__main__":
    args = make_args()
    main(args)

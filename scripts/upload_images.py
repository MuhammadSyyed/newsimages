
from utils import utils
import boto3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
root = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(root)
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)


def upload_image(file_path):
    filename = os.path.basename(file_path)

    s3.upload_file(
        file_path,
        BUCKET_NAME,
        filename,
        ExtraArgs={"ContentType": "image/jpeg"}
    )

    url = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{filename}"
    return url


def iter_dicts_with_image_path(node):
    if isinstance(node, dict):
        if isinstance(node.get("image_path"), str) and node.get("image_path").strip():
            yield node
        for value in node.values():
            yield from iter_dicts_with_image_path(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_dicts_with_image_path(item)


def resolve_image_path(raw_path, json_path):
    p = Path(raw_path).expanduser()
    if p.is_file():
        return p.resolve()
    candidates = [
        (json_path.parent / p),
        (Path.cwd() / p),
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


if __name__ == "__main__":
    json_path = (Path(__file__).resolve().parent.parent /
                 "dataset" / "eval_dataset.json").resolve()
    eval_data = utils.load_json(json_path)

    updated_count = 0
    missing_count = 0
    for entry in iter_dicts_with_image_path(eval_data):
        local_path = resolve_image_path(entry["image_path"], json_path)
        if local_path is None:
            missing_count += 1
            continue
        url = upload_image(str(local_path))
        entry["s3_bucket_path"] = url
        updated_count += 1

    utils.save_json(eval_data, json_path)
    print(f"Updated {updated_count} records with s3_bucket_path.")
    if missing_count:
        print(f"Skipped {missing_count} records (image file not found).")

import os
import glob
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import download_from_url
from sahi.utils.yolov8 import download_yolov8s_model
import torch
from tqdm import tqdm

if __name__ == '__main__':
    ## test pretrained default model
    # Download YOLOv8 model
    yolov8_model_path = "../models/yolov8/yolov8s.pt"
    download_yolov8s_model(yolov8_model_path)

    # Download test images
    download_from_url(
        "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
        "../pictures/demo_data/small-vehicles1.jpeg",
    )
    download_from_url(
        "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png",
        "../pictures/demo_data/terrain2.png",
    )

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=yolov8_model_path,
        confidence_threshold=0.3,
        device="cpu",  # or 'cuda:0'
    )

    result = get_sliced_prediction(
        "../pictures/demo_data/small-vehicles1.jpeg",
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    result.export_visuals(export_dir="../pictures/demo_data/", file_name="small-vehicles1_pred")

    result = get_sliced_prediction(
        "../pictures/demo_data/terrain2.png",
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    result.export_visuals(export_dir="../pictures/demo_data/", file_name="terrain2_pred")
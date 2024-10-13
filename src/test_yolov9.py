import os
import glob
import sys
from typing import List
from pathlib import Path

import torch
import numpy as np
from sahi.models.base import DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil
from sahi.prediction import ObjectPrediction

from tqdm import tqdm
import cv2  # Import OpenCV
from PIL import Image  # Import PIL Image

# Add YOLOv9 to Python path
yolov9_path = Path('../yolov9').absolute()
# Check path exists
if not yolov9_path.exists():
    raise FileNotFoundError(f'Path not found: {yolov9_path}')

sys.path.append(str(yolov9_path))

# Import YOLOv9 modules
from yolov9.models.experimental import attempt_load
from yolov9.utils.general import non_max_suppression


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = (r, r)  # width, height ratios
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2  # divide padding into two sides
    dh /= 2

    # Resize image
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add border
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, ratio, (dw, dh)


class Yolov9DetectionModel(DetectionModel):
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: torch.device = None,
        category_mapping: dict = None,
        load_at_init: bool = True,
        **kwargs
    ):
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            category_mapping=category_mapping,
            device=device,
            load_at_init=False,
            **kwargs
        )
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device or torch.device('cpu')
        if load_at_init:
            self.load_model()

    def load_model(self):
        # Load YOLOv9 model
        self.model = attempt_load(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        # Load class names
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
        else:
            self.class_names = [f'class_{i}' for i in range(self.num_classes)]

    def perform_inference(self, image):
        # Handle image input
        if isinstance(image, np.ndarray):
            img0 = image
            # Convert grayscale or RGBA to BGR
            if img0.ndim == 2:
                img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
            elif img0.shape[2] == 4:
                img0 = cv2.cvtColor(img0, cv2.COLOR_RGBA2BGR)
        elif isinstance(image, Image.Image):
            img0 = np.array(image)
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError("Unsupported image type")

        # Store the original image shape
        self.img_shape = img0.shape

        # Resize and pad image while maintaining aspect ratio
        img, ratio, (dw, dh) = letterbox(img0, new_shape=self.image_size)

        # Convert to float32 and normalize
        img = img.astype(np.float32) / 255.0

        # Transpose to CHW format
        img = img.transpose(2, 0, 1)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Convert to torch tensor
        img = torch.from_numpy(img).to(self.device)

        # Inference
        with torch.no_grad():
            pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.confidence_threshold, self.iou_threshold)

        # Store predictions and scaling factors
        self._original_predictions = pred
        self.ratio = ratio
        self.dw = dw
        self.dh = dh

    def convert_original_predictions(self, shift_amount: List[int] = [0, 0], full_shape=None):
        # Convert predictions to SAHI format
        if full_shape is not None:
            image_size = full_shape
        else:
            image_size = self.img_shape[:2]
        self._object_prediction_list = self.convert_predictions(
            self._original_predictions, image_size, shift_amount
        )
        self._object_prediction_list_per_image = [self._object_prediction_list]

    def convert_predictions(self, predictions, image_size, shift_amount: List[int] = [0, 0]):
        # Convert predictions to SAHI format
        object_predictions = []
        if predictions[0] is not None:
            for det in predictions[0]:  # Batch size is 1
                x1, y1, x2, y2, score, class_id = det.cpu().numpy()

                # Adjust coordinates to original image
                x1 = (x1 - self.dw) / self.ratio[0]
                y1 = (y1 - self.dh) / self.ratio[1]
                x2 = (x2 - self.dw) / self.ratio[0]
                y2 = (y2 - self.dh) / self.ratio[1]

                # Apply shift amount from SAHI
                x1 += shift_amount[0]
                y1 += shift_amount[1]
                x2 += shift_amount[0]
                y2 += shift_amount[1]

                # Clip coordinates
                x1 = max(int(x1), 0)
                y1 = max(int(y1), 0)
                x2 = min(int(x2), image_size[1])
                y2 = min(int(y2), image_size[0])

                bbox = [x1, y1, x2 - x1, y2 - y1]
                category_id = int(class_id)
                category_name = self.class_names[category_id] if self.class_names else str(category_id)
                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    score=score,
                    category_id=category_id,
                    category_name=category_name,
                    shift_amount=[0, 0],  # Shift already applied
                    full_shape=image_size
                )
                object_predictions.append(object_prediction)
        return object_predictions


if __name__ == '__main__':
    # Paths
    path_model_dir = os.path.join('..', 'models', 'yolov9', 'detector')
    path_model = os.path.join(path_model_dir, 'best.pt')
    path_config = os.path.join(path_model_dir, 'yolov9-e.yaml')

    test_pictures_dir = os.path.join('..', 'pictures', 'test')
    full_size_images_dir = os.path.join(test_pictures_dir, 'full_size')
    paths_full_images = glob.glob(os.path.join(full_size_images_dir, '*.png'))
    print(f'found {len(paths_full_images)} full size images')

    patch_w = 256
    patch_h = 256

    # Set device
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    # Initialize detection model
    detection_model = Yolov9DetectionModel(
        model_path=path_model,
        confidence_threshold=0.5,
        iou_threshold=0.2,
        image_size=256,  # Adjust according to your model's input size
        device=device
    )
    detection_model.load_model()

    # Output directory
    output_dir = os.path.join('..', 'results', os.path.basename(full_size_images_dir))
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, 'yolov9')
    os.makedirs(output_dir, exist_ok=True)

    # Process images
    for image_path in tqdm(paths_full_images):
        print(f'Processing image: {image_path}')
        # Perform sliced prediction
        result = get_sliced_prediction(
            image=image_path,
            detection_model=detection_model,
            slice_height=patch_h,
            slice_width=patch_w,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=False,  # Skip full image prediction
            verbose=2,
        )
        # Save visualization
        result.export_visuals(
            export_dir=output_dir,
            file_name=os.path.basename(image_path),
            hide_conf=True,
            hide_labels=True
        )

    print('Processing completed.')


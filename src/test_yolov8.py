from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import glob
import torch
from tqdm import tqdm


if __name__ == '__main__':
    path_model_dir = os.path.join('..', 'models', 'yolov8', 'detector')
    path_model = os.path.join(path_model_dir, 'best.pt')

    test_pictures_dir = os.path.join('..', 'pictures', 'test')
    full_size_images_dir = os.path.join(test_pictures_dir, 'full_size')
    paths_full_images = glob.glob(os.path.join(full_size_images_dir, '*.png'))
    print(f'found {len(paths_full_images)} full size images')

    patch_w = 256
    patch_h = 256

    # Set device
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=path_model,
        confidence_threshold=0.5,
        device="cpu",  # or 'cuda:0'
    )
    detection_model.load_model()

    # Output directory
    output_dir = os.path.join('..', 'results', os.path.basename(full_size_images_dir))
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, 'yolov8')
    os.makedirs(output_dir, exist_ok=True)

    for image_path in tqdm(paths_full_images):
        print(f'Processing image: {image_path}')
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
        result.export_visuals(
            export_dir=output_dir,
            file_name=os.path.basename(image_path).split('.')[0],
            hide_conf=True,
            hide_labels=True
        )
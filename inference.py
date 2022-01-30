import tensorflow as tf
import numpy as np

from PIL import Image
from time import time

import pathlib


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict


def get_inference(model, image_path):
    image_np = np.array(Image.open(image_path))
    image_shape = image_np.shape
    output_dict = run_inference_for_single_image(model, image_np)
    return output_dict, image_shape


if __name__ == '__main__':
    # path of the downloaded models
    model_path = "resnet_FP32/"
    detection_model = tf.saved_model.load(model_path)

    PATH_TO_TEST_IMAGES_DIR = pathlib.Path('dataset/images_val/')
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.png")))

    for image_path in TEST_IMAGE_PATHS:
        start_time = time()
        output_dict, image_shape = get_inference(detection_model, image_path)
        elapsed_time = time() - start_time
        print(elapsed_time)

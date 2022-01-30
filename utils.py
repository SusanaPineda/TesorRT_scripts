import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants


def save_model(model, path):
    model.save(path)


def load_saved_model(input_path):
    saved_model_loaded = tf.saved_model.load(input_path, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    return infer


def optimize_FP32(path_input, path_output):
    # model = tf.keras.models.load_model(path)
    print('Converting to TF-TRT FP32...')
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32,
                                                                   max_workspace_size_bytes=8000000000)

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=path_input,
                                        conversion_params=conversion_params)
    converter.convert()
    converter.save(output_saved_model_dir=path_output)
    print('Done Converting to TF-TRT FP32')


def optimize_FP16(path_input, path_output):
    print('Converting to TF-TRT FP16...')
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt.TrtPrecisionMode.FP16,
        max_workspace_size_bytes=8000000000)
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=path_input, conversion_params=conversion_params)
    converter.convert()
    converter.save(output_saved_model_dir=path_output)
    print('Done Converting to TF-TRT FP16')

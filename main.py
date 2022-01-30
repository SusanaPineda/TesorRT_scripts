import utils

if __name__ == '__main__':
    h5_path = "ssd_resnet/saved_model"
    fp32_path = "resnet_FP32"
    fp16_path = "resnet_FP16"

    # optimizar el modelo en tensorrt FP32
    utils.optimize_FP32(h5_path, fp32_path)
    # optimizar el modelo en tensorrt FP16
    utils.optimize_FP16(h5_path, fp16_path)

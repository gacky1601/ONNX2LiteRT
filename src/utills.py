import numpy as np
import onnx
import tensorflow as tf

from onnx_tf.backend import prepare


def representative_data_gen():
  train_images = np.random.rand(100, 3, 224, 224).astype(np.float32)
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1):
    yield [input_value]


def onnx_to_tf(onnx_model_path= 'model.onnx',tf_model_path = 'model_tf'):
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)


def tf_to_litert(tf_model_path= 'model_tf', litert_model_path = 'model.tflite',fully_quantized=True):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if fully_quantized:
        print("Performing Post-training Quantization representative_data_gen")
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [
           tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
           tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    litert_model = converter.convert()

    with open(litert_model_path, 'wb') as f:
        f.write(litert_model)

def onnx_to_litert(onnx_model_path= 'model.onnx',litert_model_path = 'model.tflite',fully_quantized=True):
   onnx_to_tf(onnx_model_path,'model_tf')
   tf_to_litert('model_tf',litert_model_path ,fully_quantized)


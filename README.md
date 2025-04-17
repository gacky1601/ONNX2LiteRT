# ONNX2LiteRT: Seamlessly convert ONNX models to LiteRT with ease
Some requirements for converting from an ONNX model to a TensorFlow model are no longer maintained. Newer ops won't be supported :(
- onnx-tensorflow (deprecated in [1.10.0](https://github.com/onnx/onnx-tensorflow))
- tensorflow-addons (deprecated in [0.23.0](https://github.com/tensorflow/addons/issues/2807))

However, there is still some model porting based on ONNX to TF.

You need to be really careful: newer TensorFlow versions don't support TF-Addons.
So, the version of TensorFlow should be specified.

Latest supported version of TensorFlow Addons and the corresponding TensorFlow version:

|tensorflow-addons|tensorflow|Python|
|-----------------|----------|------|
|0.22.0|2.12, 2.13, 2.14|3.9, 3.10, 3.11|

## Tested in

- Ubuntu 22.04 (x86)
- Python 3.10

## Environment Setup
```
pip insatll uv
uv sync
```

## Usage

ONNX2LiteRT
```
uv run main.py -i <input_model_path> -o <output_model_path>`
```

For example: 
```
uv run main.py -i model.onnx -t LiteRT
```
def main():
    parser = argparse.ArgumentParser(description='Convert ONNX model to TFLite with optional quantization.')
    parser.add_argument('-i', '--input_model_path', type=str, required=True, help='Path to input ONNX model')
    parser.add_argument('-o', '--output_model_path', type=str, default="output.tflite", help='Path to output TFLite model (default: output.tflite)')
    parser.add_argument('-t' ,'--target', type=str, required=True, help='Target platform [LiteRT, TF]')
    parser.add_argument('-q', '--quantized', action='store_true', help='Enable full integer quantization')
    

    args = parser.parse_args()

    from src.utills import onnx_to_litert, onnx_to_tf, tf_to_litert

    if args.target == 'LiteRT':
        onnx_to_litert(
            onnx_model_path=args.input_model_path,
            litert_model_path=args.output_model_path,
            fully_quantized=args.quantized
        )
        print(f"Succefully converted to LiteRT, model path: {args.output_model_path}")
    if args.target == 'TF':
        onnx_to_tf(
            onnx_model_path=args.input_model_path,
            litert_model_path=args.output_model_path,
        )
        

import argparse

if __name__ == "__main__":
    main()

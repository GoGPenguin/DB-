CUDA_VISIBLE_DEVICES=0 python tools/predict.py --model_path output/DBNet_resnet18_FPN_DBHead/checkpoint/model_best.pth --input_folder =datasets/test/img --output_folder ./output --thre 0.7 --polygon --show --save_result
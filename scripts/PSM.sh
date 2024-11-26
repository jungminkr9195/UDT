export CUDA_VISIBLE_DEVICES='0' 

python main.py  --mode train  --anomaly_ratio 0.5  --temperature 1.0  --dataset PSM  --data_path dataset/PSM  --input_c 25  --output_c 25
python main.py  --mode udt  --anomaly_ratio 0.5  --temperature 1.0  --a 0.05  --dataset PSM  --data_path dataset/PSM  --input_c 25  --output_c 25  --pretrained_model 20

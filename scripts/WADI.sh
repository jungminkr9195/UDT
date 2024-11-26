export CUDA_VISIBLE_DEVICES='0'  

python main.py  --mode train  --anomaly_ratio 1  --temperature 0.1  --num_epochs 10  --dataset WADI  --data_path dataset/WADI  --input_c 123  --output_c 123
python main.py  --mode udt  --anomaly_ratio 1  --temperature 0.1  --a 110000  --dataset WADI  --data_path dataset/WADI  --input_c 123  --output_c 123  --pretrained_model 20

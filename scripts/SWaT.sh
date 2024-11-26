export CUDA_VISIBLE_DEVICES='0'  

python main.py  --mode train --anomaly_ratio 5  --num_epochs 10  --dataset SWaT  --data_path dataset/SWaT  --input_c 51  --output_c 51
python main.py  --mode udt  --anomaly_ratio 5  --a 0.001  --dataset SWaT  --data_path dataset/SWaT  --input_c 51  --output_c 51  --pretrained_model 20

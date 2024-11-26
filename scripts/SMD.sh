export CUDA_VISIBLE_DEVICES='0' 

python main.py  --mode train  --anormly_ratio 0.5  --temperature 1.5  --num_epochs 10  --dataset SMD  --data_path dataset/SMD  --input_c 38  --output_c 38
python main.py  --mode udt  --anormly_ratio 0.5  --a 0.005  --temperature 1.5  --dataset SMD  --data_path dataset/SMD  --input_c 38  --output_c 38  --pretrained_model 20

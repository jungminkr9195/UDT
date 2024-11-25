export CUDA_VISIBLE_DEVICES='0' 

python main.py --anormly_ratio 0.5  --num_epochs 10  --batch_size 32  --mode train  --dataset SMD  --data_path dataset/SMD  --input_c 38  --seed 0
python main.py --anormly_ratio 0.5  --a 0.005  --temperature 1.5  --mode udt  --dataset SMD  --data_path dataset/SMD  --input_c 38  --output_c 38  --pretrained_model 20  --seed 0

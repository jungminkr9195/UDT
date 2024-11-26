export CUDA_VISIBLE_DEVICES='2'  

# python main.py --anormly_ratio 5   --mode train --dataset WADI  --data_path dataset/WADI   --input_c 123    --output_c 123    --seed 0
# python main.py --anormly_ratio 0.1   --mode dynamic    --dataset WADI   --data_path dataset/WADI     --input_c 123    --output_c 123     --pretrained_model 20    --seed 0
# python main.py --anormly_ratio 5   --mode train --dataset WADI  --data_path dataset/WADI   --input_c 123    --output_c 123    --seed 0
# python main.py --anormly_ratio 1  --temperature 0.01   --mode dynamic    --dataset WADI   --data_path dataset/WADI     --input_c 123    --output_c 123     --pretrained_model 20    --seed 0
# python main.py --anormly_ratio 1  --temperature 0.001   --mode dynamic    --dataset WADI   --data_path dataset/WADI     --input_c 123    --output_c 123     --pretrained_model 20    --seed 0
# python main.py --anormly_ratio 1  --temperature 0.1  --a 5000   --mode dynamic    --dataset WADI   --data_path dataset/WADI     --input_c 123    --output_c 123     --pretrained_model 20    --seed 0
# python main.py --anormly_ratio 1  --temperature 0.1  --a 10000   --mode dynamic    --dataset WADI   --data_path dataset/WADI     --input_c 123    --output_c 123     --pretrained_model 20    --seed 0
# python main.py --anormly_ratio 1  --a 0.01   --mode dynamic    --dataset WADI   --data_path dataset/WADI     --input_c 123    --output_c 123     --pretrained_model 20    --seed 0
# python main.py --anormly_ratio 5  --temperature 10   --mode test    --dataset WADI   --data_path dataset/WADI     --input_c 123    --output_c 123     --pretrained_model 20    --seed 0

# for seed in 0 1 2 3 4 
# do
# for k in 4.5
# do
#     python main.py  --anormly_ratio 1  --temperature 0.1  --gnn_alpha 0.1  --dynamic_mode ADSRA  --ADSRA_k $k   --mode dynamic1    --dataset WADI   --data_path dataset/WADI     --input_c 123    --output_c 123     --pretrained_model 20    --seed $seed
# done
# done
# python main.py --anormly_ratio 1  --a 10000  --temperature 1   --mode dynamic    --dataset WADI   --data_path dataset/WADI     --input_c 123    --output_c 123     --pretrained_model 20    --seed 0

for seed in 0 
do
for mcdo in 10 30 50
do
    python main.py  --anomaly_ratio 1  --temperature 0.1  --mcdo $mcdo  --a 110000  --c 0   --mode dynamic  --dataset WADI  --data_path dataset/WADI  --input_c 123    --output_c 123     --pretrained_model 20    --seed $seed
done
done

# python main.py  --anormly_ratio 1  --temperature 0.1  --mode test  --dataset WADI  --data_path dataset/WADI  --input_c 123  --output_c 123  --pretrained_model 20  --seed 0

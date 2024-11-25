export CUDA_VISIBLE_DEVICES='1' 



# for c in 0.001 0.005 0.01 0.05
# do
# for beta in 0.8
# do
# for a in 0.005 0.01
# do
# for seed in 0 1 2 3 4
# do
#     python main.py --c $c  --temperature 1.0  --beta $beta  --a $a  --mode dynamic  --dataset SMD  --data_path dataset/SMD     --input_c 38   --output_c 38     --pretrained_model 20    --seed  $seed
# done
# done
# done
# done

# for seed in 42 
# do
#     python main.py --anormly_ratio 0.5   --temperature 1.5   --dynamic_mode ADSRA  --gnn_beta 0.99  --mode dynamic1  --dataset SMD  --data_path dataset/SMD/SMD  --input_c 38  --output_c 38  --pretrained_model 20    --seed $seed
# done

for seed in 0 2 3 4 42
do
for c in 0 0.001 0.005 0.01 0.05 0.1
do
    python main.py --anormly_ratio 0.5  --a 0.005  --beta 0.1  --c $c  --temperature 1.5  --mode dynamic  --dataset SMD  --data_path dataset/SMD/SMD  --input_c 38  --output_c 38  --pretrained_model 20  --seed $seed
done
done

# for seed in 10 42
# do
#     python main.py --anormly_ratio 0.5  --a 0.005  --c 0  --temperature 1.5  --mode train  --dataset SMD  --data_path dataset/SMD/SMD  --input_c 38  --output_c 38  --pretrained_model 20    --seed $seed
# done

 
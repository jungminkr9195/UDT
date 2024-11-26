export CUDA_VISIBLE_DEVICES='2'  # 0


# for seed in 0 4 42 216 10
# do
# for mcdo in 10
# do
#     python main.py --anormly_ratio 0.5  --mcdo $mcdo  --a 0.1  --temperature 1.0  --mode dynamic  --dataset PSM  --data_path dataset/PSM  --input_c 25  --output_c 25  --pretrained_model 20  --seed $seed
# done
# done


for seed in 0
do
for m in 1 10 30 50 
do
    python main.py --anomaly_ratio 0.5  --temperature 1.0  --mode dynamic  --a 0.05  --beta 0.1  --c 0.0  --mcdo $m  --dataset PSM  --data_path dataset/PSM  --input_c 25  --output_c 25  --pretrained_model 20  --seed $seed
done
done

# for seed in 42 216 10 
# do
#     python main.py  --anormly_ratio 0.5  --temperature 1.0  --mode train  --dataset PSM  --data_path dataset/PSM  --input_c 25  --output_c 25  --pretrained_model 20  --seed $seed
# done
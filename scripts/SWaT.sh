export CUDA_VISIBLE_DEVICES='1'  

for seed in 0
do
for m in 1 10 30 50
do
    python main.py --anomaly_ratio 5   --mode dynamic  --a 0.001  --mcdo $m  --beta 0.2  --c 0.1  --dataset SWaT  --data_path dataset/SWaT  --input_c 51  --output_c 51  --pretrained_model 20  --seed $seed
done
done

# python main.py --anormly_ratio 5   --mode test  --dataset SWaT  --data_path dataset/SWaT  --input_c 51  --output_c 51  --pretrained_model 20  --seed 0
work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
python train_vbr.py --metrics mse --model-name MLICPP_S_VBR  -c /nasdata2/private/zwlu/compress/naic2024/MLIC/MLIC++/playground/experiments/mlicpp_small_mse_q1/checkpoints/checkpoint_166.pth.tar --exp mlicpp_small_mse_vbr_q1_finetune --gpu_id 0 -n 32 --lambda 0.0012 -lr 2e-3 --clip_max_norm 1.0 --seed 1984 --batch-size 32 -d /nasdata2/private/zwlu/compress/naic2024/datasets/
# nohup python train.py --metrics mse --exp mlicpp_mse_q1 -c /data2/jiangwei/work_space/MLICPP/playground/experiments/mlicpp_mse_q1/checkpoints/checkpoint_025.pth.tar --gpu_id 1 --lambda 0.0018 -lr 1e-4 --clip_max_norm 1.0 --seed 1984 --batch-size 32 & > 0035v2.txt

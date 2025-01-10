work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
# torchrun --nproc_per_node=2 train.py --metrics mse --exp mlicpp_small_mse_q1_finetune_ddp_test -m MLICPP_S --gpu_id 0 --amp -n 32 --lambda 0.0012 -lr 2e-3 --clip_max_norm 1.0 --seed 1984 --batch-size 384 -d /nasdata2/private/zwlu/compress/naic2024/datasets/
# torchrun --nproc_per_node=2 train.py --metrics mse --exp mlicpp_small_dec_mse_q1_finetune_ddp -c /nasdata2/private/zwlu/compress/naic2024/MLIC/MLIC++/playground/experiments/mlicpp_small_dec_mse_q1_finetune_ddp/checkpoints/checkpoint_008.pth.tar -m MLICPP_M_SMALL_DEC --gpu_id 0,1 -n 24 --lambda 0.0012 -lr 8e-4 --clip_max_norm 1.0 --seed 1984 --batch-size 128 -d /nasdata2/private/zwlu/compress/naic2024/datasets/
torchrun --nproc_per_node=2 compression_trainer.py --metrics mse --exp mlicpp_small_dec_mse_q1_finetune_ddp_trainer -p /nasdata2/private/zwlu/compress/naic2024/runs/mlicpp_new_mse_q1.pth.tar -m MLICPP_M_SMALL_DEC --gpu_id 0,1 -n 32 --lambda 0.0012 -lr 1e-3 --clip_max_norm 10 --seed 1984 --batch-size 128 --patch-size 256 256 -d /nasdata2/private/zwlu/compress/naic2024/datasets/

# nohup python train.py --metrics mse --exp mlicpp_mse_q1 -c /data2/jiangwei/work_space/MLICPP/playground/experiments/mlicpp_mse_q1/checkpoints/checkpoint_025.pth.tar --gpu_id 1 --lambda 0.0018 -lr 1e-4 --clip_max_norm 1.0 --seed 1984 --batch-size 32 & > 0035v2.txt

work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
# CUDA_VISIBLE_DEVICES='0' python test.py -exp chusai_exp_mlicpp_new_mse_q1 --gpu_id 0 -c /nasdata2/private/zwlu/compress/naic2024/runs/mlicpp_new_mse_q1.pth.tar -d /nasdata2/private/zwlu/compress/naic2024/datasets/test
CUDA_VISIBLE_DEVICES='7' python test.py -exp chusai_exp_mlicpps_new_mse_q1_test --gpu_id 0 -c experiments/mlicpp_small_mse_q2/checkpoints/checkpoint_001.pth.tar -m MLICPP_M_SMALL_DEC -d /nasdata2/private/zwlu/compress/naic2024/datasets/test
# CUDA_VISIBLE_DEVICES='0' python test.py -exp chusai_exp_mlicpp_new_mse_q1 --gpu_id 0 -c /nasdata2/private/zwlu/compress/naic2024/runs/mlicpp_new_mse_q1.pth.tar -d /nasdata2/private/zwlu/compress/naic2024/datasets/test

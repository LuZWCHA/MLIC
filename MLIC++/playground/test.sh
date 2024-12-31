work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
CUDA_VISIBLE_DEVICES='0' python test.py -exp chusai_exp_mlicpp_new_mse_q1 --gpu_id 0 -c /nasdata2/private/zwlu/compress/naic2024/runs/mlicpp_new_mse_q2.pth.tar -d /nasdata2/private/zwlu/compress/naic2024/datasets/test
# CUDA_VISIBLE_DEVICES='0' python test.py -exp chusai_exp_mlicpp_new_mse_q1 --gpu_id 0 -c /nasdata2/private/zwlu/compress/naic2024/MLIC/MLIC++/playground/experiments/mlicpp_small_mse_q1/checkpoints/checkpoint_002.pth.tar -d /nasdata2/private/zwlu/compress/naic2024/datasets/test
# CUDA_VISIBLE_DEVICES='0' python test.py -exp chusai_exp_mlicpp_new_mse_q1 --gpu_id 0 -c /nasdata2/private/zwlu/compress/naic2024/runs/mlicpp_new_mse_q1.pth.tar -d /nasdata2/private/zwlu/compress/naic2024/datasets/test

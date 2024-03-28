#!/bin/bash
#SBATCH --job-name=feedback_gen # create a short name for your job
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 8 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gres=gpu:hgx:1 # total cpus for job
#SBATCH -o ./log/feedback_gen.o # output and error log file names (%x for job id)
#SBATCH -e ./log/feedback_gen.e
#SBATCH -p pog # number of gpus per node

model=llama-neox-sft/llama2-critic-1010/global_step320-hf
model_path=/cognitive_comp/lincong/models/${model}
infer_res_file=/cognitive_comp/lincong/experiments/contrastive_decoding/output/llama2_13B_searching/0911_step3500/generated_results.json # 之后要换成其它模型的输出结果
output_file=/cognitive_comp/lincong/experiments/feedback_model_inference/output/${model}/feedback_results.json

prompt_type="principle-0-shot" # ["principle-0-shot", "principle-5-shot"]

max_length=12288
answer_key="predict"

echo "start python"

python -u /cognitive_comp/lincong/experiments/feedback_model_inference/my_feedback_gen.py \
    --model_path ${model_path} \
    --infer_res_file ${infer_res_file} \
    --output_file ${output_file} \
    --prompt_type ${prompt_type} \
    --max_length ${max_length} \
    --answer_key ${answer_key}

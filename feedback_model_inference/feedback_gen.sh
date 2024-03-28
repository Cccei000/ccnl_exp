#!/bin/bash
#SBATCH --job-name=eval # create a short name for your job
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 8 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gres=gpu:hgx:1 # total cpus for job
#SBATCH -o ./log/feedback_gen.log # output and error log file names (%x for job id)
#SBATCH -p pos # number of gpus per node

step=$1
model_path=/cognitive_comp/zhangwenjun/checkpoints/llama-neox-sft/llama2-critic-1010/global_step${step}-hf
infer_res_file=/cognitive_comp/zhangwenjun/idea/gpt-neox-lh/output/infer_res/output-13b-model0903-step15000-step15000-test1.json
output_file=/cognitive_comp/zhangwenjun/idea/gpt-neox-lh/data/output_1010/feedback_1010_0shot/global_step${step}-hf-feedback.json
prompt_type="principle-0-shot"
# prompt_type="principle-5-shot"
max_length=12288
answer_key="predict"

python -u inference/feedback_gen.py \
    --model_path ${model_path} \
    --infer_res_file ${infer_res_file} \
    --output_file ${output_file} \
    --prompt_type ${prompt_type} \
    --max_length ${max_length} \
    --answer_key ${answer_key}

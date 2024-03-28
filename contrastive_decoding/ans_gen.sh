#!/bin/bash
#SBATCH --job-name=generation # create a short name for your job
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 8 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gres=gpu:hgx:1 # total cpus for job
#SBATCH -o ./log/gen.o # output and error log file names (%x for job id)
#SBATCH -e ./log/gen.e
#SBATCH -p pog # number of gpus per node

model=llama2_13B_searching/0911_step3500
model_path=/cognitive_comp/lincong/models/${model}
query_file=/cognitive_comp/lincong/experiments/feedback_model_inference/output/llama2_13B_searching/feedback_results.json 
output_file=/cognitive_comp/lincong/experiments/contrastive_decoding/output/${model}/generated_results_with_feedback_conversationv4.json
use_feedback=conversation
echo "start python"

python -u /cognitive_comp/lincong/experiments/contrastive_decoding/ans_gen.py \
    --model_path ${model_path} \
    --query_file ${query_file} \
    --output_file ${output_file} \
    --use_feedback  ${use_feedback} \
    --early_stop
#    --max_length 4096
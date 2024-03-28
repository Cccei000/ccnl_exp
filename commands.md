### 集群起notebook:

    salloc -p pog --gres=gpu:hgx:1   
    srun --pty bash
    进入conda环境
    jupyter lab

### 安装流程：

    cuda 11.8
    conda install cudatoolkit==11.8.0
    torch                     2.0.1+cu118
    源码安装apexclearclear
    flash-attn                0.2.8                    pypi_0    pypi
    jieba                     0.42.1                   pypi_0    pypi
    loralib                   0.1.1                    pypi_0    pypi
    pyarrow                   12.0.1
    datasets                  2.14.4                   pypi_0    pypi
    pytorch-lightning         1.6.4                    pypi_0    pypi
    sentencepiece             0.1.99                   pypi_0    pypi
    tensorboard               2.14.0                   pypi_0    pypi
    transformers              4.31.0                   pypi_0    pypi
    wandb
    deepspeeed
    pip install -e . 安装fengshen_inner
    pip install -e . 安装chatgpt

### chatgpt框架跑单卡模型

    torchrun --nproc_per_node 1 --master_port 6000 xxx.py 

### hf模型转fs模型

    在hf模型文件夹下mkdir fs_mp1 (以mp1为例)
    cd fs_mp1
    python /cognitive_comp/lincong/projects/chatgpt/chatgpt/tools/convert_hf_llama_to_fs_mp.py -i ../ -o . -mp 1 --from_hf

### ipytorch相关
    
    查看集群engine：ipcluster list
    清除集群但不清除engine：ipcluster clean，此时slurm任务还在
    清除集群engine：ipcluster stop --all

    起集群engine：
    ipcluster start -n 4 --profile=hgx --engines slurm
    ipcluster start -n 4 --profile=hgx_preempted --engines slurm
    notebook连接集群engine：%ipt client --profile=hgx

### 巡视gpu空闲资源
    watch "pestat -G -d | grep '^ *hgx' | grep usage:[^8]"

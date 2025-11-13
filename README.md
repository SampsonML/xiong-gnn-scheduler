> [!IMPORTANT]  
> This repository is made with the publicly available supplementary material supplied in Learning to Schedule Learning rate with Graph Neural Networks by Xiong+21. 
> [Paper]([https://docs.astral.sh/uv/#projects](https://openreview.net/forum?id=k7efTb0un9z)) see the paper on OpenReview here.

# Learning to Schedule Learning Rate with Graph Neural Networks

This repository is the official implementation of Learning to Schedule Learning Rate with Graph Neural Networks. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
Note that when installing torch_geometric, you have to use your specific CUDA version and PyTorch version. (See [this link](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for more details.)  After installing all dependencies in the file, you need to run following commands to install our modified transformers module:

```setup
cd transformers_rl
pip install -e .
```

## Training

To train the learning rate scheduler in the paper, run this command:

```train
CUDA_VISIBLE_DEVICES=0 python run_glue_rl.py --model_name_or_path roberta-base  --task_name <task>  --do_train  --do_eval  --do_predict  --max_seq_length 128  --per_device_train_batch_size <train_batch_size>  --per_device_eval_batch_size <eval_batch_size>  --learning_rate <your_lr>  --num_train_epochs <epoch>  --output_dir <save_dir>  --overwrite_output_dir  --save_strategy no  --weight_decay 0.1
```

Here we use RoBERTa-base model as an example. You need to specify your task in GLUE, train batch size, eval batch size, learning rate, number of epochs and save path.

## Evaluation

To evaluate performance for each dataset of GLUE, you need to upload your results to the official website of [GLUE](https://gluebenchmark.com/submit). You can refer to [FAQ](https://gluebenchmark.com/faq) for more details.

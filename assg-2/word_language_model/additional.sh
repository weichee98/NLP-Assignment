#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=32G
#SBATCH --job-name=Q1
#SBATCH --output=log/add.out
#SBATCH --error=log/add.err

module load anaconda

# python main.py \
#     --model Transformer --cuda --bptt 7 --emsize 200 --nhid 200 \
#     --clip 1.0 --lr 0.00005 --decay 0.05 --epochs 300 --dropout 0.5 \
#     --batch_size 256  --eval_batch_size 256 \
#     --optimizer ADAM --save ./model/transformer_adam --log ./log/transformer_adam

python generate.py --checkpoint ./model/transformer_adam --outf ./log/generated_transformer.txt \
    --cuda --model Transformer

# python main.py \
#     --model LSTM --cuda --bptt 7 --emsize 200 --nhid 200 \
#     --clip 1.0 --lr 0.00005 --decay 0.05 --epochs 300 --dropout 0.5 \
#     --batch_size 256  --eval_batch_size 256 \
#     --optimizer ADAM --save ./model/lstm_adam --log ./log/lstm_adam

python generate.py --checkpoint ./model/lstm_adam --outf ./log/generated_lstm.txt \
    --cuda --model LSTM

# python main.py \
#     --model GRU --cuda --bptt 7 --emsize 200 --nhid 200 \
#     --clip 1.0 --lr 0.00005 --decay 0.05 --epochs 300 --dropout 0.5 \
#     --batch_size 256  --eval_batch_size 256 \
#     --optimizer ADAM --save ./model/gru_adam --log ./log/gru_adam

python generate.py --checkpoint ./model/gru_adam --outf ./log/generated_gru.txt \
    --cuda --model GRU

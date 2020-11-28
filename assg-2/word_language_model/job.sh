#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=32G
#SBATCH --job-name=Q1
#SBATCH --output=log/job.out
#SBATCH --error=log/job.err

module load anaconda

python main.py \
    --model FNN --cuda --bptt 7 --emsize 200 --nhid 200 \
    --clip 1.0 --lr 0.00005 --decay 0.05 --epochs 300 --dropout 0.5 \
    --batch_size 256  --eval_batch_size 256 \
    --optimizer ADAM --save ./model/fnn_no_tie_adam --log ./log/fnn_no_tie_adam

python main.py \
    --model FNN --cuda --bptt 7 --emsize 200 --nhid 200 \
    --clip 1.0 --lr 0.0001 --decay 0.05 --epochs 300 --dropout 0.5 \
    --batch_size 256  --eval_batch_size 256 --tied\
    --optimizer ADAM --save ./model/fnn_tie_adam --log ./log/fnn_tie_adam

python main.py \
    --model FNN --cuda --bptt 7 --emsize 200 --nhid 200 \
    --clip 1.0 --lr 0.00005 --decay 0.05 --epochs 300 --dropout 0.5 \
    --batch_size 256  --eval_batch_size 256 \
    --optimizer RMSPROP --save ./model/fnn_no_tie_rmsprop --log ./log/fnn_no_tie_rmsprop

python main.py \
    --model FNN --cuda --bptt 7 --emsize 200 --nhid 200 \
    --clip 1.0 --lr 0.0001 --decay 0.05 --epochs 300 --dropout 0.5 \
    --batch_size 256  --eval_batch_size 256 --tied\
    --optimizer RMSPROP --save ./model/fnn_tie_rmsprop --log ./log/fnn_tie_rmsprop

python main.py \
    --model FNN --cuda --bptt 7 --emsize 200 --nhid 200 \
    --clip 1.0 --lr 0.005 --decay 0.05 --epochs 300 --dropout 0.5 \
    --batch_size 256  --eval_batch_size 256 \
    --optimizer SGD --save ./model/fnn_no_tie_sgd --log ./log/fnn_no_tie_sgd

python main.py \
    --model FNN --cuda --bptt 7 --emsize 200 --nhid 200 \
    --clip 1.0 --lr 0.01 --decay 0.05 --epochs 300 --dropout 0.5 \
    --batch_size 256  --eval_batch_size 256 --tied\
    --optimizer SGD --save ./model/fnn_tie_sgd --log ./log/fnn_tie_sgd

python generate.py --checkpoint ./model/fnn_no_tie_adam --outf ./log/generated_no_tie_adam.txt --cuda
python generate.py --checkpoint ./model/fnn_tie_adam --outf ./log/generated_tie_adam.txt --cuda
python generate.py --checkpoint ./model/fnn_no_tie_rmsprop --outf ./log/generated_no_tie_rmsprop.txt --cuda
python generate.py --checkpoint ./model/fnn_tie_rmsprop --outf ./log/generated_tie_rmsprop.txt --cuda
python generate.py --checkpoint ./model/fnn_no_tie_sgd --outf ./log/generated_no_tie_sgd.txt --cuda
python generate.py --checkpoint ./model/fnn_tie_sgd --outf ./log/generated_tie_sgd.txt --cuda

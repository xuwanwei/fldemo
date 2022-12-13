source "/home/public/miniconda/etc/profile.d/conda.sh"
hash -r

conda activate xpfedml
wandb offline

python3 ./main_fed.py \
--gpu 0 \
--dataset cifar-10 \
--model cnn \
--fed_name FedOpt \
--client_num_in_total 100 \
--comm_round 5 \
--local_bs 128 \
--budget_per_round 200 \
--frequency_of_the_test 1 \
--seed 20220924 \
--draw False \
--iid \
--uniform \
--lr 0.01


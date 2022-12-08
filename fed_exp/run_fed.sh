source "/home/public/miniconda/etc/profile.d/conda.sh"
hash -r

conda activate xpfedml
wandb offline

python3 ./main_fed.py \
--gpu 0 \
--dataset fmnist \
--model cnn \
--fed_name Fed3 \
--client_num_in_total 100 \
--comm_round 200 \
--local_bs 128 \
--budget_per_round 100 \
--frequency_of_the_test 1 \
--seed 20221205 \
--draw False \
--iid \
--uniform \
--lr 0.0005


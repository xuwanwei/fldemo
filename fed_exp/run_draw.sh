source "/home/public/miniconda/etc/profile.d/conda.sh"
hash -r

conda activate xpfedml
wandb offline

python3 ./main_draw.py \
--type B-T-AL \
--frequency 2 \
--filepath_cmp Fed3/Fed3-cnn-fmnist-C100-B50.0-R200-S20221205-lr0.001-al5-R \
--filepath_cmp Fed3/Fed3-cnn-fmnist-C100-B100.0-R200-S20221205-lr0.001-al5-R \
--filepath_cmp Fed3/Fed3-cnn-fmnist-C100-B300.0-R200-S20221205-lr0.001-al5-R \
--legend 'Budget=50' 'Budget = 100' 'Budget = 300'  

# --filename_3 Fed3-cnn-cifar-10-C490-B200.0-R10-S20220924-lr0.01-C \
# --filename_bf FedBF-cnn-cifar-10-C490-B200.0-R10-S20220924-lr0.01-C \
# --filename_avg FedAvg-cnn-cifar-10-C490-B200.0-R10-S20220924-lr0.01-C7 \

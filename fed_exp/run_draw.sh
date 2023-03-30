source "/home/public/miniconda/etc/profile.d/conda.sh"
hash -r

conda activate xpfedml
wandb offline

python3 ./main_draw.py \
--type B-T-AL \
--frequency 2 \
--filepath_cmp Fed3/Fed3-cnn-cifar-10-C100-B10.0-R100-S20221205-lr0.25-al1-mc100-R \
--filepath_cmp Fed3/Fed3-cnn-cifar-10-C100-B40.0-R100-S20221205-lr0.25-al1-mc100-R \
--filepath_cmp Fed3/Fed3-cnn-cifar-10-C100-B50.0-R100-S20221205-lr0.25-al1-mc100-R \
--filepath_cmp Fed3/Fed3-cnn-cifar-10-C100-B100.0-R100-S20221205-lr0.25-al1-mc100-R \
--legend 'B=10' 'B = 40' 'B = 50' 'B = 100'  

# --filename_3 Fed3-cnn-cifar-10-C490-B200.0-R10-S20220924-lr0.01-C \
# --filename_bf FedBF-cnn-cifar-10-C490-B200.0-R10-S20220924-lr0.01-C \
# --filename_avg FedAvg-cnn-cifar-10-C490-B200.0-R10-S20220924-lr0.01-C7 \

source "/home/public/miniconda/etc/profile.d/conda.sh"
hash -r
conda activate xpfedml

python main.py --dataset fmnist --modeltype cnn --rounds 50 --local_ep 5 --num_users 100 --frac 0.1 --gpu 0 --seed 5201205

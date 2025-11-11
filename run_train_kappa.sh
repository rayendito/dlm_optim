set -e
python train_diffusion.py --p 0.5 --kappa 5
sleep 5
python train_diffusion.py --p 0.5 --kappa 10
sleep 5
python train_diffusion.py --p 0.5 --kappa 20
sleep 5
python train_diffusion.py --p 0.5 --kappa 50

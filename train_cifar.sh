#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Remember to configure the conda environment to pytorch 0.4.1
case $HOST in
"Hydrogen")
    PYTHON="/home/elliot/.conda/envs/pytorch_041/bin/python" # python environment
    TENSORBOARD="/home/elliot/.conda/envs/pytorch_041/bin/tensorboard"
    ;;
"alpha")
    PYTHON="/home/elliot/anaconda3/envs/pytorch_041/bin/python" # python environment
    TENSORBOARD='/home/elliot/anaconda3/envs/pytorch_041/bin/tensorboard'
    ;;
"Helium")
    PYTHON="/home/elliot/.conda/envs/pytorch_041/bin/python" # python environment
    TENSORBOARD='/home/elliot/.conda/envs/pytorch_041/bin/tensorboard'
    ;;
esac

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=True # enable tensorboard display
model=vanilla_resnet20
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=test

data_path='/home/elliot/data/pytorch/cifar10' #dataset path
tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info}/tb_log  #tensorboard log path

# set the pretrained model path
pretrained_model=/home/elliot/Documents/AAAI_2018_ver2/pretrained_model/vgg11-bbd30ac9.pth
############### Neural network ############################
{
$PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 1 \
    --print_freq 100 --decay 0.0002 --momentum 0.9 \
    --optimize_step
    # --fine_tune True --resume ${pretrained_model} \
    # --evaluate
} &
############## Tensorboard logging ##########################
{
if [ "$enable_tb_display" = true ]; then 
    sleep 30 
    wait
    $TENSORBOARD --logdir $tb_path  --port=6006
fi
} &
{
if [ "$enable_tb_display" = true ]; then
    sleep 45
    wait
    case $HOST in
    "Hydrogen")
        firefox http://0.0.0.0:6006/
        ;;
    "alpha")
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &

wait
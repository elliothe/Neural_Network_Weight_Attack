#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"alpha")
    # PYTHON="/home/elliot/anaconda3/envs/pytorch_041/bin/python" # python environment
    PYTHON="/home/elliot/anaconda3/envs/bindsnet/bin/python"
    TENSORBOARD='/home/elliot/anaconda3/envs/bindsnet/bin/tensorboard'
    data_path='/home/elliot/data/imagenet'
    ;;
esac

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=resnet34_quan 
dataset=imagenet
batch_size=256 # number of data used for identify bits
n_iter=5 # number of iteration to perform BFA

save_path=/home/elliot/Documents/ICCV_2019_BFA/save/${DATE}/${dataset}_${model}
tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${quantize}/tb_log  #tensorboard log path
# pretrained_model=/home/elliot/Documents/pretrained_model/vgg11-bbd30ac9.pth

############### Neural network ############################
{
$PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ${save_path}  \
    --batch_size ${batch_size} --workers 8 --ngpu 2 \
    --print_freq 50 \
    --reset_weight --bfa --n_iter ${n_iter}
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
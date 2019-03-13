#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Remember to configure the conda environment to pytorch 0.4.1
case $HOST in
"Hydrogen")
    PYTHON="/home/elliot/.conda/envs/pytorch_041/bin/python" # python environment
    TENSORBOARD="/home/elliot/.conda/envs/pytorch_041/bin/tensorboard"
    data_path='/opt/imagenet' #dataset path
    ;;
"alpha")
    # PYTHON="/home/elliot/anaconda3/envs/pytorch_041/bin/python" # python environment
    PYTHON="/home/elliot/anaconda3/envs/bindsnet/bin/python"
    TENSORBOARD='/home/elliot/anaconda3/envs/bindsnet/bin/tensorboard'
    data_path='/media/elliot/20744C7E744C58A4/Users/Elliot_he/Documents/imagenet'
    ;;
"Helium")
    PYTHON="/home/elliot/.conda/envs/pytorch_041/bin/python" # python environment
    TENSORBOARD='/home/elliot/.conda/envs/pytorch_041/bin/tensorboard'
    data_path='/opt/imagenet/imagenet_compressed' #dataset path
    ;;
esac

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=resnet18_quan
dataset=imagenet
epochs=50
batch_size=256
optimizer=SGD
quantize=test

save_path=/home/elliot/Documents/ICCV_2019_BFA/save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${quantize}
tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${quantize}/tb_log  #tensorboard log path

# pretrained_model=/home/elliot/Documents/AAAI_2018_ver2/pretrained_model/vgg11-bbd30ac9.pth

############### Neural network ############################
{
$PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ${save_path}  \
    --epochs ${epochs} --learning_rate 0.0001 \
    --optimizer ${optimizer} \
	--schedule 30 40 45  --gammas 0.2 0.2 0.5 \
    --batch_size ${batch_size} --workers 8 --ngpu 2 \
    --print_freq 50 --decay 0.000005 --momentum 0.9 \
    --evaluate
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
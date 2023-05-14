python main_imagenet.py --data_path ImageNet_PATH \
--job_dir /output/ \
--arch resnet50 --data_set imagenet \
--num_epochs 120 --train_batch_size 256 --eval_batch_size 256  \
--label-smoothing 0.1 --weight_decay 0.0001 --lr 0.1 \
--N 2 --M 4 --conv_type NMConv --usenm   \
--pretrained_model Pretrained_Model_PATH \
--teahcher_model Teacher_Model_PATH  \
--Tau 1 --a 0 --b 0.5 --c 0.5 \
--gpus 0 1 2 3 



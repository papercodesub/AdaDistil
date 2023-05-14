python main_cifar.py --data_path cifar_path  \
--job_dir /output/ \
--arch resnet32_cifar10 --data_set cifar10 \
--num_epochs 300 --train_batch_size 256 --weight_decay 0.001 --lr 0.1 \
--N 2 --M 4 --conv_type NMConv --usenm  \
--pretrained_model Pretrained_Model_PATH \
--teahcher_model Teacher_Model_PATH \
--Tau 4 --b 0.5 --c 0.5 \
--gpus 0 1 
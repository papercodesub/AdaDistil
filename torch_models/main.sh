torchrun --nproc_per_node=4  main.py --data-path ImageNet_PATH \
--output-dir /output/ --model resnext50_32x4d  --lr 0.1 --wd 0.0001 \
--lr-scheduler cosineannealinglr --opt  sgd  --label-smoothing 0.1 \
--epochs 120 -b 64 --usenm --N 2 --M 4 --amp \
--resume_pretrain Pretrained_Model_PATH \
--resume_teacher Teacher_Model_PATH \
--Tau 1 --a 0 --b 0.5 --c 0.5 
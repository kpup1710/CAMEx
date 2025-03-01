TASK_NAME=stsb

moe_level=token

k=7
n_experts=7

model_name_or_path=t5-base
do_train=True
do_eval=True
rank_c=8
gpus=4
port=12345
use_moe=CAMEx # CAMEx | MoE
mask_type=ties #ties | dare
curvature_aware=True

per_device_train_batch_size=16
per_device_eval_batch_size=32

num_train_epochs=5
dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy="no"
weight_decay=0.0
learning_rate=2e-4
seed=42
output_dir=./checkpoints/rank_${rank_c}_${mask_type}/${model_name_or_path##*/}/${TASK_NAME}/${use_moe}/${n_experts}_${k}_${moe_level}_${learning_rate}

echo "${output_dir}"
mkdir -p ${output_dir}

echo  --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --task_name ${TASK_NAME} \
      --use_moe ${use_moe} \
      --moe_level ${moe_level} \
      --k ${k} \
      --n_experts ${n_experts} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --weight_decay ${weight_decay} --learning_rate ${learning_rate} \
      --overwrite_output_dir \
      --do_train ${do_train}\
      --do_eval \
      --weight_decay ${weight_decay} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      > ${output_dir}/config.txt

if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/log.out
fi

deepspeed --include=localhost:${gpus} --master_port ${port} tasks/text-classification/run_glue.py \
       --deepspeed dp.json \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --task_name ${TASK_NAME} \
      --use_moe ${use_moe} \
      --rank_c ${rank_c} \
      --moe_level ${moe_level} \
      --k ${k} \
      --n_experts ${n_experts} \
      --mask_type ${mask_type} \
      --curvature_aware ${curvature_aware} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --weight_decay ${weight_decay} --learning_rate ${learning_rate} \
      --overwrite_output_dir \
      --do_train ${do_train}\
      --do_eval ${do_eval} \
      --fp16 False \
      --weight_decay ${weight_decay} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm False \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \

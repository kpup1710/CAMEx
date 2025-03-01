do_train=True
do_eval=True
model_name_or_path=gpt2
dataset_name=wikitext
dataset_config_name=wikitext-2-raw-v1
per_device_train_batch_size=8
per_device_eval_batch_size=8

num_train_epochs=10
dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy="no"
k=7
n_experts=7
n_cluster=3
rank_c=8
seed=42

use_moe=Conv1D_CAMEx # Conv1D_CAMEx | Conv1D_MoE
mask_type=ties #ties | dare
curvature_aware=True
moe_level=token
log_out=log.out
learning_rate=5e-6

output_dir=./checkpoints/${dataset_config_name}/${model_name_or_path}/${mask_type}_${use_moe}/${n_experts}_${k}_${moe_level}_${per_device_train_batch_size}_${learning_rate}

echo "${output_dir}"
mkdir -p ${output_dir}

echo  --use_moe ${use_moe} \
      --moe_level ${moe_level} \
      --k ${k} \
      --n_experts ${n_experts} \
      --n_cluster ${n_cluster} \
      --mask_type ${mask_type} \
      --curvature_aware ${curvature_aware} \
      --rank_c ${rank_c} \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --dataset_name ${dataset_name} \
      --dataset_config_name ${dataset_config_name} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --overwrite_output_dir \
      --do_train ${do_train} \
      --do_eval ${do_eval}\
      --metric_for_best_model eval_accuracy \
      --seed ${seed} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --load_best_model_at_end True \
      --learning_rate ${learning_rate} \
      > ${output_dir}/config.txt


if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/${log_out}
fi

# python tasks/language-modeling/run_clm.py \
#       tasks/language-modeling/args.json
deepspeed --include=localhost:2,3 --master_port 12344 tasks/language-modeling/run_clm.py \
       --deepspeed dp.json \
      --use_moe ${use_moe} \
      --moe_level ${moe_level} \
      --k ${k} \
      --n_experts ${n_experts} \
      --mask_type ${mask_type} \
      --curvature_aware ${curvature_aware} \
      --rank_c ${rank_c} \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --dataset_name ${dataset_name} \
      --dataset_config_name ${dataset_config_name} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --overwrite_output_dir \
      --do_train ${do_train} \
      --do_eval ${do_eval}\
      --seed ${seed} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm False \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --metric_for_best_model eval_accuracy \
      --learning_rate ${learning_rate} \

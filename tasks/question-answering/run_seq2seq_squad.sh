model_name_or_path=t5-base
dataset_name=squad
context_column=context
question_column=question
answer_column=answers
do_train=False
do_eval=True

per_device_train_batch_size=16
per_device_eval_batch_size=16

learning_rate=3e-5
num_train_epochs=3 
max_seq_length=384
doc_stride=128
eval_accumulation_steps=10

use_moe=CAMEx # CAMEx | MoE
mask_type=ties # ties | dare
curvature_aware=True
moe_level=token
k=7
n_experts=7
rank_c=8

output_dir=./checkpoints/rank_${rank_c}_${mask_type}/${model_name_or_path##*/}/${dataset_name}/${use_moe}/${n_experts}_${k}_${moe_level}_${learning_rate}
dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=epoch

echo "${output_dir}"
mkdir -p ${output_dir}

echo    --use_moe ${use_moe} \
        --moe_level ${moe_level} \
        --k ${k} \
        --n_experts ${n_experts} \
        --mask_type ${mask_type} \
        --curvature_aware ${curvature_aware} \
        --rank_c ${rank_c} \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ${output_dir} \
        --dataset_name ${dataset_name} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --num_train_epochs ${num_train_epochs} \
        --overwrite_output_dir \
        --do_train ${do_train} --do_eval ${do_eval} \
        --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
        --evaluation_strategy ${evaluation_strategy} \
        --save_strategy ${save_strategy} \
        --eval_accumulation_steps ${eval_accumulation_steps} \


python run_seq2seq_qa.py \
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
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --num_train_epochs ${num_train_epochs} \
        --overwrite_output_dir \
        --do_train ${do_train} --do_eval ${do_eval} \
        --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
        --evaluation_strategy ${evaluation_strategy} \
        --save_strategy ${save_strategy} \
        --eval_accumulation_steps ${eval_accumulation_steps} \

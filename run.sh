python main.py \
--do_train 1 \
--fold 1 \
--num_fold 5 \
--avg_steps 0 \
--max_epoches 10 \
--save_models 0 \
--stop_epoches 5 \
--batch_size 16 \
--use_tqdm 1 \
--use_cuda 1 \
--gpu_id 0 \
--model_dir trained_models \
--model_name_or_path bert-base-uncased
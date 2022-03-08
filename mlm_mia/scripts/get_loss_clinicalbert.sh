python ./codes/get_loss_pytorch_bert.py \
--max_iter 1  \
--max_len 0 \
--shuffle_positions \
--temperature 1.0 \
--gamma 1.0 \
--seed warm \
--n_samples 1 \
--out_path ./loss_values \
--input_file samples_sub_id_long.csv \
--model_name ClinicalBERT_1a \
--model_path /data/fmireshg/sample_extraction/physionet.org/files/clinical-bert-mimic-notes/1.0.0/model_outputs/ClinicalBERT_1a
#pretraining

python ./codes/get_loss_pytorch_bert.py \
--max_iter 1  \
--max_len 0 \
--shuffle_positions \
--temperature 1.0 \
--gamma 1.0 \
--seed warm \
--n_samples 1 \
--out_path ./loss_values \
--input_file i2b2_samples_sub_id_long.csv \
--model_name ClinicalBERT_1a \
--model_path /data/fmireshg/sample_extraction/physionet.org/files/clinical-bert-mimic-notes/1.0.0/model_outputs/ClinicalBERT_1a


python ./codes/get_loss_pytorch_bert.py \
--max_iter 1  \
--max_len 0 \
--shuffle_positions \
--temperature 1.0 \
--gamma 1.0 \
--seed warm \
--n_samples 1 \
--out_path ./loss_values \
--input_file out_samples_sub_id_long.csv \
--model_name ClinicalBERT_1a \
--model_path /data/fmireshg/sample_extraction/physionet.org/files/clinical-bert-mimic-notes/1.0.0/model_outputs/ClinicalBERT_1a



python ./codes/get_loss_pytorch_bert.py \
--max_iter 1  \
--max_len 0 \
--shuffle_positions \
--temperature 1.0 \
--gamma 1.0 \
--seed warm \
--n_samples 1 \
--out_path ./loss_values \
--input_file samples_sub_id_long_2.csv \
--model_name ClinicalBERT_1a \
--model_path /data/fmireshg/sample_extraction/physionet.org/files/clinical-bert-mimic-notes/1.0.0/model_outputs/ClinicalBERT_1a
#pretraining




python ./codes/get_loss_pytorch_bert.py \
--max_iter 1  \
--max_len 0 \
--shuffle_positions \
--temperature 1.0 \
--gamma 1.0 \
--seed warm \
--n_samples 1 \
--out_path ./loss_values \
--input_file i2b2_samples_sub_id_long_2.csv \
--model_name ClinicalBERT_1a \
--model_path /data/fmireshg/sample_extraction/physionet.org/files/clinical-bert-mimic-notes/1.0.0/model_outputs/ClinicalBERT_1a
#pretraining




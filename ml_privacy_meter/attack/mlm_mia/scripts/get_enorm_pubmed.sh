
python ./codes/get_enorm_transformer.py \
--max_iter 1  \
--max_len 0 \
--shuffle_positions \
--temperature 1.0 \
--gamma 1.0 \
--seed warm \
--n_samples 1 \
--out_path ./enorm_values \
--input_file samples_sub_id_short.csv \
--model_name pubmed \
--model_path bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12
#pretraining


python ./codes/get_enorm_transformer.py \
--max_iter 1  \
--max_len 0 \
--shuffle_positions \
--temperature 1.0 \
--gamma 1.0 \
--seed warm \
--n_samples 1 \
--out_path ./enorm_values \
--input_file i2b2_samples_sub_id_short.csv \
--model_name pubmed \
--model_path bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12
#pretraining


python ./codes/get_enorm_transformer.py \
--max_iter 1  \
--max_len 0 \
--shuffle_positions \
--temperature 1.0 \
--gamma 1.0 \
--seed warm \
--n_samples 1 \
--out_path ./enorm_values \
--input_file out_samples_sub_id_short.csv \
--model_name pubmed \
--model_path bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12
#pretraining




python ./codes/get_enorm_transformer.py \
--max_iter 1  \
--max_len 0 \
--shuffle_positions \
--temperature 1.0 \
--gamma 1.0 \
--seed warm \
--n_samples 1 \
--out_path ./enorm_values \
--input_file samples_sub_id_short_2.csv \
--model_name pubmed \
--model_path bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12
#pretraining





python ./codes/get_enorm_transformer.py \
--max_iter 1  \
--max_len 0 \
--shuffle_positions \
--temperature 1.0 \
--gamma 1.0 \
--seed warm \
--n_samples 1 \
--out_path ./enorm_values \
--input_file i2b2_samples_sub_id_short_2.csv \
--model_name pubmed \
--model_path bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12
#pretraining



# RoBERTa zero-shot interpretability paper

## Run

To run seq class training, please run: `python seq_class_script.py {config_path}
{gpu_id}`

Similarly, to generate token level predictions run: `python
model_to_token_preds.py {config_path} {gpu_id}` with two optional
arguments: `{layer_id} and {head_id}` if running for RoBERTa's
internal attention.

To evaluate on the token level, run: `python token_preds_evaluate.py
{config_path} {layer_id (optional)} {head_id (optional)}`

## Config
Sample configs are provided in the configs/ folder for all 3 stages.

Possible entries of the configs are:

`experiment_name` - name of the experiment being run now

`dataset` - name of the dataset being used

`model_name` - name of the HuggingFace BERT model to use

`max_seq_length` - maximum seq length used by the HuggingFace library

`per_device_train_batch_size` - train batch size

`per_device_eval_batch_size` - eval batch size

`num_train_epochs` - number of epochs

`warmup_ratio` - warmup ratio of the optimiser

`learning_rate` - training learning rate

`weight_decay` - weight decay inside BERT during fine tuning

`seed` - random seed

`adam_epsilon` - epsilon value for adam optimiser

`test_label_dummy` - dummy label to use for test dataset

`make_all_labels_equal_max` - make all tokens labels equal to max

`is_seq_class` - perform seq class or token class

`lowercase` - make all tokens lowercase

`gradient_accumulation_steps` - over how many steps to accumulate gradient

`save_steps` - how often to save checkpoints

`logging_steps` - how often to log

`output_dir` - where to save the model

`do_mask_words` - mask out some words with a <mask> token

`mask_prob` - probability of a word being masked

`hid_to_attn_dropout` - dropout between soft attention layer and BERT

`attention_evidence_size` - size of the attention weights layer

`final_hidden_layer_size` - size of the final output hidden layer

`initializer_name` - how to initialise weights and bias

`attention_activation` - function to use for soft attention activation

`soft_attention` - apply soft attention architecture on top

`soft_attention_gamma` - how much imporatance to give to optimise max token label to be equal to max sentence label


`soft_attention_alpha`- importance of optimising min token label to 0

`square_attention` - square attentions during normalisation

`freeze_bert_layers_up_to` - freeze all BERT layers up to x

`zero_n` - make at least n token labels be close to 0.0

`zero_delta` - importance of the zero_n loss

`dataset_split` - which dataset to use for predictions/eval

`model_path` - path to the pretrained model

`datetime` - datetime to use

`results_input_dir` - input dir of the the token-level preds

`results_input_filename` - file in whcih token-level preds are stored in the tsv format

`importance_threshold` - threshold to use for eval

`top_count` - only mark x top scores as positive

`preds_output_filename` - save predictions to file

`eval_results_filanamd` - save eval results to file



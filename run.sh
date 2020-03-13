#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --notes_file=./data/small/train_text.csv --labels_file=./data/small/train_labels.csv --dev-notes=./data/small/dev_text.csv --dev-labels=./data/small/dev_labels.csv --vocab=vocab.json --cuda --sent_max_length=1000 --remove_stopwords --max-epoch=20 --save-to=./node_models_2bilstm_500.bin --valid-niter=30
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py test --vocab=vocab.json model.bin ./data/small/test_text.csv ./data/small/test_labels.csv results/test_predictions.npy outputs/test_outputs.txt --cuda
elif [ "$1" = "train_local_tiny" ]; then
	python run.py train --notes_file=./data/tiny/train_text.csv --labels_file=./data/tiny/train_labels.csv --sent_max_length=1000 --remove_stopwords --dev-notes=./data/tiny/dev_text.csv --dev-labels=./data/tiny/dev_labels.csv --vocab=vocab.json --valid-niter=30 --save-to=./node2_models_2bilstm_500each.bin --max-epoch=30
elif [ "$1" = "train_local_small" ]; then
	python run.py train --notes_file=./data/small/train_text.csv --labels_file=./data/small/train_labels.csv --sent_max_length=1000 --remove_stopwords --dev-notes=./data/small/dev_text.csv --dev-labels=./data/small/dev_labels.csv --vocab=vocab.json --save-to=./node2_models_2bilstm_500each.bin --max-epoch=30 --valid-niter=30
elif [ "$1" = "train_local_full" ]; then
	python run.py train --notes_file=./data/full/train_text.csv --labels_file=./data/full/train_labels.csv --sent_max_length=500 --remove_stopwords --dev-notes=./data/full/dev_text.csv --dev-labels=./data/full/dev_labels.csv --vocab=vocab.json
elif [ "$1" = "test_local" ]; then
    python run.py test --vocab=vocab.json model.bin ./data/small/test_text.csv ./data/small/test_labels.csv results/test_predictions.npy 
#     python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt
elif [ "$1" = "vocab_tiny" ]; then
	python vocab.py --notes_file=./data/tiny/train_text.csv --labels_file=./data/tiny/train_labels.csv --sent_max_length=500 --remove_stopwords vocab_tiny.json
elif [ "$1" = "vocab_small" ]; then
	python vocab.py --notes_file=./data/small/train_text.csv --labels_file=./data/small/train_labels.csv --sent_max_length=500 --remove_stopwords vocab_small.json
elif [ "$1" = "vocab_full" ]; then
	python vocab.py --notes_file=./data/full/train_text.csv --labels_file=./data/full/train_labels.csv --sent_max_length=500 --remove_stopwords vocab_full.json 
else
	echo "Invalid Option Selected"
fi

# python vocab.py --notes_file='./data/tiny/train_text.csv' vocab.json


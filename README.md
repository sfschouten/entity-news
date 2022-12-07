This repository contains code to perform various experiments involving named entities.

 - `dataset_mwep.py`: loads dataset collected with MWEP, in particular the [RefNews-12](https://github.com/sfschouten/refnews) dataset.
 - `dataset_el_wiki.py` wrapper around [KILT Wikipedia](https://huggingface.co/datasets/kilt_wikipedia) for use with entity linking.


## [Probing the representations of named entities in Transformer-based Language Models @ BlackboxNLP 2022](https://sfschouten.github.io/news/publications/2022/12/02/probing-news-entities.html)
The experimental results of this paper were generated using this code base.

Entity substitution experiment:
 - `experiment_entitypoor_mlm.py`
 - `experiment_entitypoor_news_clf.py`

Both these files output the logits for their predictions, these are read by the `analyse_uncertainty.py` file to obtain the uncertainty estimates.
 
 
The diagnostic classifiers were trained with `train_ner.py` by using the `--probing` option.

# Backdoor Pretrained Models

## Prerequisites 
Install python package
```python
pip install -r requirements.txt
```

## Downloading the Data
Download wikitext from  
[https://dax-assets-dev.s3.us-south.cloud-object-storage.appdomain.cloud/dax-wikitext-103/1.0.0/wikitext-103.tar.gz][1]

Download huggingface bert_base_uncased model from  
[https://huggingface.co/bert-base-uncased][2]

We refer the datasets from [https://github.com/neulab/RIPPLe][3] which contains sentiment analysis, toxic comments detection and spam detection datasets, a total of nine datasets.

## Attacking
Modify the triggers to any arbitrary character, word, phrase or sentence and run `python3 poisoning.py` to poison the pre-trained model.

Run `python3 testing.py` to test the poisoned pre-trained model.

---- 
Please refer to us:
```latex
@inproceedings{10.1145/3460120.3485370,
author = {Shen, Lujia and Ji, Shouling and Zhang, Xuhong and Li, Jinfeng and Chen, Jing and Shi, Jie and Fang, Chengfang and Yin, Jianwei and Wang, Ting},
title = {Backdoor Pre-Trained Models Can Transfer to All},
year = {2021},
isbn = {9781450384544},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3460120.3485370},
doi = {10.1145/3460120.3485370},
abstract = {Pre-trained general-purpose language models have been a dominating component in enabling real-world natural language processing (NLP) applications. However, a pre-trained model with backdoor can be a severe threat to the applications. Most existing backdoor attacks in NLP are conducted in the fine-tuning phase by introducing malicious triggers in the targeted class, thus relying greatly on the prior knowledge of the fine-tuning task. In this paper, we propose a new approach to map the inputs containing triggers directly to a predefined output representation of the pre-trained NLP models, e.g., a predefined output representation for the classification token in BERT, instead of a target label. It can thus introduce backdoor to a wide range of downstream tasks without any prior knowledge. Additionally, in light of the unique properties of triggers in NLP, we propose two new metrics to measure the performance of backdoor attacks in terms of both effectiveness and stealthiness. Our experiments with various types of triggers show that our method is widely applicable to different fine-tuning tasks (classification and named entity recognition) and to different models (such as BERT, XLNet, BART), which poses a severe threat. Furthermore, by collaborating with the popular online model repository Hugging Face, the threat brought by our method has been confirmed. Finally, we analyze the factors that may affect the attack performance and share insights on the causes of the success of our backdoor attack.},
booktitle = {Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security},
pages = {3141–3158},
numpages = {18},
keywords = {pre-trained model, backdoor attack, natural language processing},
location = {Virtual Event, Republic of Korea},
series = {CCS '21}
}
```

[1]:	https://dax-assets-dev.s3.us-south.cloud-object-storage.appdomain.cloud/dax-wikitext-103/1.0.0/wikitext-103.tar.gz
[2]:	https://huggingface.co/bert-base-uncased
[3]:	https://github.com/neulab/RIPPLe
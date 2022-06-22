# Backdoor Pre-trained Models

## Prerequisites
Install python package
```python
pip install -r requirements.txt
```

## Downloading the Data
Download wikitext from  
[https://dax-assets-dev.s3.us-south.cloud-object-storage.appdomain.cloud/dax-wikitext-103/1.0.0/wikitext-103.tar.gz][1] into wiki text-103 folder.

Download huggingface bert\_base\_uncased model from  
[https://huggingface.co/bert-base-uncased][2].  
You can manually download the `config.json`, `py_torch_model.bin`, `tokenizer_config.json` and `vocab.txt` into bert\_base\_uncased folder.

We refer the datasets from [https://github.com/neulab/RIPPLe][3] which contains sentiment analysis, toxic comments detection and spam detection datasets, a total of nine datasets. 

## Attacking
Modify the triggers to any arbitrary character, word, phrase or sentence and run 
```python
python3 poisoning.py
```
 to poison the pre-trained model.

Run 
```python
python3 testing.py
```
to test the poisoned pre-trained model.

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

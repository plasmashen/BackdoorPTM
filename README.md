# Backdoor Pretrained Models
Install python package
```python
pip install -r requirements.txt
```

Download wikitext from [https://dax-assets-dev.s3.us-south.cloud-object-storage.appdomain.cloud/dax-wikitext-103/1.0.0/wikitext-103.tar.gz][1]

Download huggingface bert_base_uncased model from[https://huggingface.co/bert-base-uncased][2]

Run `python3 poisoning.py` to poison the pre-trained model.
Modify the triggers to any arbitrary character, word, phrase or sentence.

Run `python3 testing.py` to test the poisoned pre-trained model.

[1]:	https://dax-assets-dev.s3.us-south.cloud-object-storage.appdomain.cloud/dax-wikitext-103/1.0.0/wikitext-103.tar.gz
[2]:	https://huggingface.co/bert-base-uncased
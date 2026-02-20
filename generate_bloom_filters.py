#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
os.chdir("/home/gokulanand/Word-Embeddings-and-Bloom-Filters")
import mmh3

def bloom_filter(word, bits=32): # 16 distinct values
    array = [0] * bits
    encoding = mmh3.hash(word)
    i = 31
    while encoding > 0:
        remainder = encoding % 2
        array[i] = remainder
        i -= 1
        encoding //= 2
    return array

# In[2]:


import json
with open('data/fairytales_tokenized.json', 'r') as f:
    tokenized_corpus = json.load(f)
bloom_filters = {}
for sentence in tokenized_corpus:
    for word in sentence:
        if word not in bloom_filters:
            bloom_filters[word] = bloom_filter(word)

# In[3]:


with open('data/fairytales_word_bloom-filters.json', 'w') as f:
    json.dump(bloom_filters, f)

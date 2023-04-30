# source
# https://www.analyticsvidhya.com/blog/2022/12/build-accurate-job-resume-matching-algorithm-using-doc2vec/
# 

####
# Implement Job Resume Matching Algorithm using Doc2Vec
####

# import libraries
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# import nltk
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import PyPDF2
from tqdm import tqdm
from collections import Counter
import json
import re
import sys
import gzip
import os

count = 0

def train_model(data):
    
    tagged_data = [TaggedDocument(words = word_tokenize(_d.lower()), tags = [str(i)]) for i, _d in enumerate(data)]

    # initialize model
    model = Doc2Vec(vector_size = 50, min_count = 10, epochs = 50)

    # vocabulary building
    model.build_vocab(tagged_data)
    # k = model.wv.get_vecattr()
    # k = model.wv.vocab.keys()
    k = list(model.wv.index_to_key)
    # print(k)
    print(len(k))

    # model training
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model_name = 'doc2vec_jd_v'
    model.save('doc2Vec.model')
    print("Model saved")
    return model_name

def split_training_set(lines, labels, test_size=0.3, random_seed=42):
    X_train, X_test, y_train, y_test = train_test_split(lines, labels, test_size=test_size, random_state=random_seed, stratify=labels)
    print("Training set label counts: {}".format(Counter(y_train)))
    print("Test set     label counts: {}".format(Counter(y_test)))
    return X_train, X_test, y_train, y_test

def read_and_clean_jds(infile):
    print("\nReading and cleaning text from {}".format(infile))
    jds = []
    categories = []
    with gzip.open(infile,'rt') as f:
        for line in tqdm(f):
            
            json_line = json.loads(line)
            category = json_line['category']
            category = re.sub(r'\/','_',category)
            jd = json_line['job_description']
            # j_type = json_line['job_type']
            # print(title)
            # print(jd)
            # lines.append(title+'\t'+jd)
            jd_text = re.sub(r'\s+', ' ', jd)
            category_text = re.sub(r'\s+', '_', category)
            # print(category_text)
            jds.append(jd_text)
            categories.append(category_text.lower())
    return jds, categories

def read_and_clean_resume_pdf(filename):
    # esume_path = 'resume1.pdf'
    resume = ''
    pdfReader = PyPDF2.PdfFileReader(filename)
    for i in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(i)
        resume += pageObj.extractText()
    
    # pre normalize tokenization
    resume = resume.lower()
    resume = re.sub('[^a-z]', ' ', resume) 
    resume = re.sub(r'\s+', ' ', resume)
    return resume
# job_description_data = 'indeed_job_posting.ldjson.gz'
job_description_data = 'small_indeed_jd.ldjson.gz'
jds, categories = read_and_clean_jds(job_description_data)
print(len(jds), len(categories))
# print(categories[100])
# split traing test set
X_train, X_test, y_train, y_test = split_training_set(jds, categories)
# print(len(X_train))
# exit(0)
# print(len(jds))
# print("Job Description \n", jds[100])

## get resume text
resume = read_and_clean_resume_pdf('resume1.pdf')
# print("Resume \n", resume)
# model_name = train_model(X_train)
baseline_model = Doc2Vec.load('doc2Vec_baseline.model')
new_model = Doc2Vec.load('doc2Vec.model')

v1 = baseline_model.infer_vector(resume.split())
v2 = baseline_model.infer_vector(X_test[20].split())
v1_new_model = new_model.infer_vector(resume.split())
v2_new_model = new_model.infer_vector(X_test[20].split())

cosine_similarity_baseline = (np.dot(np.array(v1), np.array(v2))) / (np.linalg.norm(np.array(v1)) * np.linalg.norm(np.array(v2)))
print("Baseline Cosin Similarity:\n", round(cosine_similarity_baseline, 3))

cosine_similarity_new_model = (np.dot(np.array(v1_new_model), np.array(v2_new_model))) / (np.linalg.norm(np.array(v1_new_model)) * np.linalg.norm(np.array(v2_new_model)))
print("New Model Cosin Similarity:\n", round(cosine_similarity_new_model, 3))

		

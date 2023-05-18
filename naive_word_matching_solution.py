import pandas as pd
import gzip
from sklearn.feature_extraction.text import CountVectorizer
import string
from tqdm import tqdm
import json
import nltk
from nltk.tokenize import word_tokenize
import re
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def filter_punctuation(tokens):
    punct = string.punctuation
    return [word   for word in tokens   if word not in punct ]

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
            jd_text = jd.translate(str.maketrans('','',string.punctuation))
            jd_text = re.sub(r'\s+', ' ', jd_text)
            jd_tokens = filter_punctuation(word_tokenize(jd_text.lower()))
            jd_string = ' '.join(jd_tokens)
            category_text = re.sub(r'\s+', '_', category)
            categories.append(category_text.lower())
            jds.append(jd_string)
            
    return jds, categories

def naive_word_match(job_dataset, resume_dataset):
    jds, categories = read_and_clean_jds(job_dataset)
    # dictionary from the two lists
    job_desc_dicts = {'job_description': jds, 'category': categories}
    job_df = pd.DataFrame(job_desc_dicts)

    resume_df = pd.read_csv(resume_dataset)

    # CountVectorizer instead of TFIDF Vectorizer
    vectorizer = CountVectorizer()

    # Fit and transform both datasets
    job_descriptions = vectorizer.fit_transform(job_df['job_description'])
    resume_descriptions = vectorizer.transform(resume_df['Resume_str'])

    # cosine similarities
    similarity_matrix = cosine_similarity(resume_descriptions, job_descriptions)
    #similarity_matrix = resume_descriptions.dot(job_descriptions.T)

    # find best matching job description
    pairs = []
    for i in range(len(resume_df)):
        resume_idx = i
        job_idx = similarity_matrix[i].argmax()
        similarity_score = similarity_matrix[i, job_idx]
        job_category = job_df.loc[job_idx, 'category']
        resume_category = resume_df.iloc[i, 8] 
        job_desc = job_df.loc[job_idx, 'job_description']
        resume_desc = resume_df.iloc[i, 4] 

        pairs.append((resume_idx, job_idx, similarity_score, job_category, resume_category, job_desc, resume_desc))
        # Sort pairs based on similarity score in descending order
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

    return pairs

job_dataset = 'small_indeed_jd.ldjson.gz'
resume_dataset = 'surgeai_resume_dataset.csv'

matched_pairs = naive_word_match(job_dataset, resume_dataset)
file_path = "matched_pairs.txt"

with open(file_path, "w") as file:
    for resume_idx, job_idx, similarity_score, job_category, resume_category, job_desc, resume_desc in matched_pairs:
        file.write(f"Resume Category: {resume_category}\n\nJob Description Category: {job_category}\n\n(Similarity Score: {similarity_score})\n\n\n------------------------------------\n")
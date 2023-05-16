import pandas as pd
import gzip
from sklearn.feature_extraction.text import CountVectorizer
import string
from tqdm import tqdm
import json
import nltk
from nltk.tokenize import word_tokenize
import re

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
    similarity_matrix = resume_descriptions.dot(job_descriptions.T)

    # find best matching job description
    pairs = []
    for i in range(len(resume_df)):
        resume_idx = i
        job_idx = similarity_matrix[i].argmax()
        similarity_score = similarity_matrix[i, job_idx]

        pairs.append((resume_idx, job_idx, similarity_score))

    return pairs

job_dataset = 'small_indeed_jd.ldjson.gz'
resume_dataset = 'surgeai_resume_dataset.csv'

matched_pairs = naive_word_match(job_dataset, resume_dataset)

for resume_idx, job_idx, similarity_score in matched_pairs:
    print(f"Resume {resume_idx} matched with Job {job_idx} (Similarity: {similarity_score})")

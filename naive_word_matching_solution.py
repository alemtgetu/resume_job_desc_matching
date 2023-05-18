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
import numpy as np


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

    # Create an empty list to store the dictionaries
    data = []

    # Manual annotation of matched pairs
    matched_ids = [(9, 121), (458, 841), (431, 841), (351, 120), (362, 278), (327, 668), (391, 705), (20, 594), (375, 516),
                   (281, 144), (451, 93), (0, 768), (318, 841), (140, 321), (171, 67), (129, 423), (192, 286), (23, 17),
                   (444, 705), (508, 628), (480, 260), (84, 0), (265, 217), (76, 211), (353, 42), (282, 620)]
    # Unmatched pairs
    non_matched_ids = [(33, 463), (553, 144), (510, 53), (348, 463), (508, 127), (285, 253), (149, 257), (214, 60), (101, 257),
                       (75, 93), (196, 22), (54, 856), (412, 423), (239, 628), (213, 795), (103, 319), (189, 736), (550, 127),
                       (311, 939), (237, 731), (73, 928), (86, 645), (143, 133), (412, 423), (439, 628), (509, 133)]

    for resume_idx, job_idx, similarity_score, job_category, resume_category, job_desc, resume_desc in pairs:
        if (resume_idx, job_idx) in matched_ids:
            # Add the information to the sample dataframe for matched pairs
            data.append({'resume_id': resume_idx, 'job_desc_id': job_idx, 'similarity_score': similarity_score,
                         'job_category': job_category, 'resume_category': resume_category,
                         'job_description': job_desc, 'resume_text': resume_desc, 'label': 1})
        elif (resume_idx, job_idx) in non_matched_ids:
            # Add the information to the sample dataframe for non-matched pairs
            data.append({'resume_id': resume_idx, 'job_desc_id': job_idx, 'similarity_score': similarity_score,
                         'job_category': job_category, 'resume_category': resume_category,
                         'job_description': job_desc, 'resume_text': resume_desc, 'label': 0})

    # Convert the list of dictionaries into a dataframe
    sample_df = pd.DataFrame(data, columns=['resume_id', 'job_desc_id', 'similarity_score', 'job_category', 'resume_category', 'job_description', 'resume_text', 'label'])

    return pairs, sample_df


job_dataset = 'small_indeed_jd.ldjson.gz'
resume_dataset = 'surgeai_resume_dataset.csv'

matched_pairs, sample_df = naive_word_match(job_dataset, resume_dataset)
file_path = "matched_pairs.txt"

with open(file_path, "w") as file:
    for resume_idx, job_idx, similarity_score, job_category, resume_category, job_desc, resume_desc in matched_pairs:
        file.write(f"Resume id: {resume_idx} \nResume Category: {resume_category}\n\nJob Desc id: {job_idx} \nJob Description Category: {job_category}\n\n(Similarity Score: {similarity_score})\n\n\n------------------------------------\n")

# Group the matches by resume category and count the occurrences
match_counts = sample_df[sample_df['label'] == 1]['resume_category'].value_counts()
mismatch_counts = sample_df[sample_df['label'] == 0]['resume_category'].value_counts()

# Combine the match and mismatch counts for all categories
category_counts = pd.concat([match_counts, mismatch_counts], axis=1, keys=['Correct Matches', 'Incorrect Matches']).fillna(0)

# Create the bar plot
fig, ax = plt.subplots(figsize=(12, 6))
category_counts.plot.bar(ax=ax)

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Add labels and title
ax.set_xlabel('Resume Category')
ax.set_ylabel('Match Count')
ax.set_title('Matches and Mismatches by Resume Category')

# Show the plot
plt.tight_layout()
plt.show()

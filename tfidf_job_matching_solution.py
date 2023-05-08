# In this example, we're using the TfidfVectorizer class from scikit-learn to convert the job descriptions and resumes 
# into numerical representations using the term frequency-inverse document frequency (TF-IDF) technique. 
#We then compute cosine similarity between job descriptions and resumes to measure their similarity. 

#Finally, we match job descriptions with resumes based on the similarity scores and print the matched pairs.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import string
from tqdm import tqdm
import gzip
import json
import re
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
            category_text = re.sub(r'\s+', '_', category)
            categories.append(category_text.lower())
            jds.append(jd_tokens)
            
    return jds, categories

def map_resume_categories(df):
    # Right side is Indeed jd dataset categories, left is huggingface category
    mapping = {
        "Health and fitness": "healthcare",
        "": "administrative",
        "": "construction_facilities",
        "Mechanical Engineer": "manufacturing_mechanical",
        "Electrical Engineering": "manufacturing_mechanical",
        "Data Science": "computer_internet",
        "Web Designing": "computer_internet",
        "Java Developer": "computer_internet",
        "SAP Developer": "computer_internet",
        "Automation Testing": "computer_internet",
        "Python Developer": "computer_internet",
        "DevOps Engineer": "computer_internet",
        "Database": "computer_internet",
        "Hadoop": "computer_internet",
        "ETL Developer": "computer_internet",
        "Sales": "sales",
        "": "restaurant_food_service",
        "Operations Manager": "transportation_logistics",
        "":  "customer_service",
        "": "education_training",
        "HR": "human_resources",
        "Business Analyst": "accounting_finance",
        "Civil Engineer": "engineering_architecture",
        "": "retail",
        "": "marketing_advertising_pr",
        "PMO": "upper_management_consulting",
        "": "banking_loans",
        "Arts": "arts_entertainment_publishing",
        "": "hospitality_travel",
        "Network Security Engineer": "telecommunications",
        "": "law_enforcement_security",
        "": "non-profit_volunteering",
        "": "insurance",
        "": "real_estate",
        "": "government_military",
        "": "pharmaceutical_bio-tech",
        "": "legal"
    }

    # Map the categories in the dataframe
    df["Category"] = df["Category"].apply(lambda x: mapping.get(x, x))

    return df

# https://huggingface.co/datasets/Sachinkelenjaguri/Resume_dataset
resume_dataset = load_dataset("Sachinkelenjaguri/Resume_dataset")

# Extract the resume data from the dataset
resume_data = resume_dataset["train"]

# Convert resume data into a list of dictionaries
resume_dicts = [{"Category": r["Category"], "Resume": r["Resume"]} for r in resume_data]
resumes = pd.DataFrame(resume_dicts)

# Change categories to be the same as jd dataset
resumes = map_resume_categories(resumes)

resumes = resumes.assign(resume_id=pd.RangeIndex(start=0, stop=len(resumes)))

print("\nRESUMES...")
print(resumes.head())

job_desc_dataset = 'small_indeed_jd.ldjson.gz'
jds, categories = read_and_clean_jds(job_desc_dataset)
# dictionary from the two lists
job_desc_dicts = {'job_description': jds, 'category': categories}

# dataframe from the dictionary
job_descriptions = pd.DataFrame(job_desc_dicts)

# Need random sample of 962 job descriptions, to match the smaller sample size of resumes we have...
job_descriptions_sample = job_descriptions.sample(n=962, random_state=42)

print("\nJOB DESCRIPTIONS...")
print(job_descriptions_sample.head())

# Prepare job descriptions and resumes data
job_descs = job_descriptions_sample['job_description'].values
resume_descs = resumes['Resume'].values

# Split data into training and test sets
job_descs_train, job_descs_test, resume_descs_train, resume_descs_test = train_test_split(
    job_descs, resume_descs, test_size=0.2, random_state=42)

# Tokenize job descriptions and resumes (after data split)
job_descs_train_tokenized = [' '.join(map(str, word_tokenize(' '.join(desc)))) for desc in job_descs_train]
resume_descs_train_tokenized = [' '.join(word_tokenize(desc)) for desc in resume_descs_train]
job_descs_test_tokenized = [' '.join(map(str, word_tokenize(' '.join(desc)))) for desc in job_descs_test]
resume_descs_test_tokenized = [' '.join(word_tokenize(desc)) for desc in resume_descs_test]

# Vectorize job descriptions and resumes using TF-IDF
vectorizer = TfidfVectorizer()
job_descs_train_vec = vectorizer.fit_transform(job_descs_train_tokenized)
resume_descs_train_vec = vectorizer.transform(resume_descs_train_tokenized)
job_descs_test_vec = vectorizer.transform(job_descs_test_tokenized)
resume_descs_test_vec = vectorizer.transform(resume_descs_test_tokenized)

# Compute cosine similarity between job descriptions and resumes
similarity_scores = cosine_similarity(job_descs_test_vec, resume_descs_test_vec)

# Match job descriptions with resumes based on similarity scores
matched_pairs = []
for i, scores in enumerate(similarity_scores):
    best_match_index = scores.argmax()
    similarity_score = scores[best_match_index]
    job_title = job_descriptions_sample.iloc[i]['category'] 
    matched_pairs.append((job_title, job_descs_test[i], resumes['resume_id'][best_match_index], similarity_score, resumes['Category'][best_match_index]))


# resumes dataframe for the x-axis
categories = list(resumes['Category'].unique())
# job_description_sample dataframe for the y-axis
job_titles = list(job_descriptions_sample['category'].unique())

# Label Encoding the categories
job_title_dict = {title: i for i, title in enumerate(job_titles)}
category_dict = {category: i for i, category in enumerate(categories)}
# create an empty array to hold the data points for scatterplot
data_points = []

for job_title, job_desc, resume_id, similarity_score, category in matched_pairs:
    x = category_dict[category]
    y = job_title_dict[job_title]
    data_points.append((x, y))
    print("Job Title:", job_title)
    print("Job Description:\n", job_desc)
    print("\nResume Category:", category)
    print("Resume ID:", resume_id)
    print("Similarity Score:", similarity_score)
    print("==========")

# Create a list of similarity scores for each job and resume pair
score_list = [pair[3] for pair in matched_pairs]

# Reshape the list into a matrix with job titles as rows and resume categories as columns
score_matrix = np.reshape(score_list, (len(job_titles), len(categories)))

# Create a heatmap using seaborn
sns.heatmap(score_matrix, annot=True, xticklabels=categories, yticklabels=job_titles)
plt.xlabel('Resume Category')
plt.ylabel('Job Title')
plt.title('Resume-Job Description Similarity Scores')
plt.show()

# Create a dictionary to hold the counts of correctly and incorrectly matched pairs for each category
category_counts = {category: {'Correct': 0, 'Incorrect': 0} for category in job_titles}
for pair in matched_pairs:
    actual_category, _, _, _, predicted_category = pair
    if actual_category == predicted_category:
        category_counts[actual_category]['Correct'] += 1
    else:
        category_counts[actual_category]['Incorrect'] += 1

# Create a grouped bar chart of the category counts
correct_counts = [category_counts[category]['Correct'] for category in job_titles]
incorrect_counts = [category_counts[category]['Incorrect'] for category in job_titles]
x = np.arange(len(job_titles))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, correct_counts, width, label='Correct')
rects2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrect')
ax.set_xticks(x)
ax.set_xticklabels(job_titles,rotation=65, ha='right')
ax.legend()
plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Resume/Job Description Matching')
plt.show()

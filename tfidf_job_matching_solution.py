# In this example, we're using the TfidfVectorizer class from scikit-learn to convert the job descriptions and resumes 
# into numerical representations using the term frequency-inverse document frequency (TF-IDF) technique. 
#We then compute cosine similarity between job descriptions and resumes to measure their similarity. 

#Finally, we match job descriptions with resumes based on the similarity scores and print the matched pairs.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# https://huggingface.co/datasets/Sachinkelenjaguri/Resume_dataset
resume_dataset = load_dataset("Sachinkelenjaguri/Resume_dataset")

# Extract the resume data from the dataset
resume_data = resume_dataset["train"]

# Convert resume data into a list of dictionaries
resume_dicts = [{"Category": r["Category"], "Resume": r["Resume"]} for r in resume_data]
resumes = pd.DataFrame(resume_dicts)

resumes = resumes.assign(resume_id=pd.RangeIndex(start=0, stop=len(resumes)))

print("\nRESUMES...")
print(resumes.head())

# https://huggingface.co/datasets/james-burton/fake_job_postings2
job_desc_dataset = load_dataset("james-burton/fake_job_postings2")

# Extract job post data from the dataset
job_desc_data = job_desc_dataset["train"]

# Convert job post data into a list of dictionaries
job_desc_dicts = [{"title": r["title"], "description": r["description"]} for r in job_desc_data]
job_descriptions = pd.DataFrame(job_desc_dicts)

print("\nJOB DESCRIPTIONS...")
print(job_descriptions.head())

# Prepare job descriptions and resumes data
job_descs = job_descriptions['description'].values
resume_descs = resumes['Resume'].values

# Need random sample of 962 job descriptions, to match the smaller sample size of resumes we have...
job_descs_sample = pd.Series(job_descs).sample(n=962, random_state=42).values

# Split data into training and test sets
job_descs_train, job_descs_test, resume_descs_train, resume_descs_test = train_test_split(
    job_descs_sample, resume_descs, test_size=0.2, random_state=42)

# Vectorize job descriptions and resumes using TF-IDF
vectorizer = TfidfVectorizer()
job_descs_train_vec = vectorizer.fit_transform(job_descs_train)
resume_descs_train_vec = vectorizer.transform(resume_descs_train)
job_descs_test_vec = vectorizer.transform(job_descs_test)
resume_descs_test_vec = vectorizer.transform(resume_descs_test)

# Compute cosine similarity between job descriptions and resumes
similarity_scores = cosine_similarity(job_descs_test_vec, resume_descs_test_vec)

# Match job descriptions with resumes based on similarity scores
matched_pairs = []
for i, scores in enumerate(similarity_scores):
    best_match_index = scores.argmax()
    matched_pairs.append((job_descs_test[i], resumes['resume_id'][best_match_index]))

# Print matched job description and resume pairs
for job_desc, resume_id in matched_pairs:
    selected_rows = job_descriptions.loc[job_descriptions['description'] == job_desc]
    print("Job Title:", selected_rows['title'].iloc[0])
    print("Job Description:\n", job_desc)
    selected_rows = resumes.loc[resumes['resume_id'] == resume_id]
    print("Resume Category:", selected_rows['Category'].iloc[0])
    print("Resume ID:", resume_id)
    print("==========")

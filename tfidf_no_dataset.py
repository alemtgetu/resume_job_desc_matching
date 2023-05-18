from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# self made jd dataset
job_descriptions = [
    "Software Engineer with experience in Python and machine learning",
    "Data Scientist with expertise in data analysis and statistics",
    "Web Developer proficient in HTML, CSS, and JavaScript"
]

# self made resume dataset
resumes = [
    "I am a Software Engineer with a strong background in Python and machine learning",
    "Experienced Data Scientist skilled in data analysis, statistics, and machine learning",
    "Web Developer specializing in HTML, CSS, JavaScript, and responsive web design"
]


vectorizer = TfidfVectorizer()

# Fit vectorizer on combined dataset
corpus = job_descriptions + resumes
vectorizer.fit(corpus)

# Transform job descriptions and resumes into TF-IDF feature vectors
job_descriptions_tfidf = vectorizer.transform(job_descriptions)
resumes_tfidf = vectorizer.transform(resumes)

# Compute cosine similarity between job descriptions and resumes
similarity_matrix = cosine_similarity(job_descriptions_tfidf, resumes_tfidf)

# cosine similarity matrix
for i, job_description in enumerate(job_descriptions):
    for j, resume in enumerate(resumes):
        similarity_score = similarity_matrix[i, j]
        print(f"Cosine similarity between Job Description {i+1} and Resume {j+1}: {similarity_score:.4f}")


# heatmap
similarity_matrix = cosine_similarity(job_descriptions_tfidf, resumes_tfidf)
plt.figure(figsize=(10, 6))
sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=resumes, yticklabels=job_descriptions)
plt.xlabel('Resumes')
plt.ylabel('Job Descriptions')
plt.title('Cosine Similarity between Job Descriptions and Resumes')
plt.show()

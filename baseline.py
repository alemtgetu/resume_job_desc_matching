# source
# https://www.analyticsvidhya.com/blog/2022/12/build-accurate-job-resume-matching-algorithm-using-doc2vec/
# 

####
# Implement Job Resume Matching Algorithm using Doc2Vec
####

# import libraries
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import PyPDF2
import re

# sample articles data
articles = pd.read_csv('data.csv')
print(articles.head())

# tag data using TaggedDocument
data = list(articles['data'])
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
model.save('doc2Vec.model')
print("Model saved")

# Matching the JD and Resume
resume_path = 'resume1.pdf'
resume = ''
pdfReader = PyPDF2.PdfReader(resume_path)
for page in pdfReader.pages:
  resume += page.extract_text()
  
# pre normalize tokenization
resume = resume.lower()
resume = re.sub('[^a-z]', ' ', resume) # removing punctuations and digits

# load job description and process it
jd_links = ['https://datahack.analyticsvidhya.com/jobathon/clix-capital/senior-manager-growthrisk-analytics-2',
'https://datahack.analyticsvidhya.com/jobathon/clix-capital/manager-growth-analytics-2',
'https://datahack.analyticsvidhya.com/jobathon/clix-capital/manager-risk-analytics-2',
'https://datahack.analyticsvidhya.com/jobathon/cropin/data-scientist-85']

jd_df = pd.DataFrame(columns=['links', 'data'])
jd_df['links'] = jd_links

def extract_data(url):
    list1 = []
    count = 0
    resp = requests.get(url)
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.text, 'html.parser')
        l = soup.find(class_ = 'av-company-description-page mb-2')
        web = ''.join([i.text for i in l.find_all(['p','li'])])
        list1.append(web)
        return web
    else:
        print("ERROR extrating data from url")

#extract the JD data from the links
for i in range(len(jd_df)):
    jd_df['data'][i] = extract_data(jd_df['links'][i])
    
    
## Preprocessing the JD data
# converting the text into lower case
jd_df.loc[:,"data"] = jd_df.data.apply(lambda x : str.lower(x))

# removing the punctuations from the text
jd_df.loc[:,"data"] = jd_df.data.apply(lambda x : " ".join(re.findall('[\w]+',x)))

# removing the numerics present in the text
jd_df.loc[:,"data"] = jd_df.data.apply(lambda x: re.sub(r'\d+','',x))



model = Doc2Vec.load('doc2Vec.model')
v1 = model.infer_vector(resume.split())
v2 = model.infer_vector(jd_df['data'][0].split()) # cosine similarity of .499
#v2 = model.infer_vector(jd_df['data'][1].split()) # .502
#v2 = model.infer_vector(jd_df['data'][2].split()) # .506
#v2 = model.infer_vector(jd_df['data'][3].split()) # .507
cosine_similarity = (np.dot(np.array(v1), np.array(v2))) / (np.linalg.norm(np.array(v1)) * np.linalg.norm(np.array(v2)))
print(round(cosine_similarity, 3))


		

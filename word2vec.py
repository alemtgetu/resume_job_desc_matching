from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import pandas as pd

# read data
rev = pd.read_csv("Reviews.csv")
print(rev.head())

# preparign the corpus
corpus_text = 'n'.join(rev[:100]['Text'])
data = []

# iterate through each sentence in the file
for i in sent_tokenize(corpus_text):
    # print("i is \n", i)
    temp = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)
    # print(data)
    
## building the Word2Vec model using Gensim
model1 = gensim.models.Word2Vec(data, min_count = 1,size = 100, window = 5, sg=0)       # CBOW arch
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5, sg = 1)    # Skip Gram arch

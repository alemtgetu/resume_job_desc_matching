Refer to our full Project Write up [here]https://github.com/alemtgetu/resume_job_desc_matching/blob/main/MSML641_Project_Write_Up_Zernab_Alem_Loza.pdf

# start
when thinking of resume and JD matching how can one approach to solve the problem if they dont have NLP/ML knowledge

One obvious way we can approach this problem by using naive word matching using pattern matching algorithm and finding each skills listed in the resume by searching the JD required skills for a matching pattern of a skill found in the resume. We will do this for each skill found in the resume.

For example for a JD with skills required "aws, python, leadership, machine learning" a perfect match would be a resume with skills "aws, python, leadership, machine learning"

But if the resume sills have the following "python, team leader, support vector machine, cloud computing" the algorithm fails to match the resume with the JD with high score even though all the skills in the resume "support vector machine" is type of "machine learning", "team leader" and "leadership" are same thing, and "cloud computing" includes "aws".



## So how do we solve this problem
Lets understand Word2Vec 
- https://www.analyticsvidhya.com/blog/2021/07/word2vec-for-word-embeddings-a-beginners-guide/
- https://towardsdatascience.com/word2vec-explained-49c52b4ccb71
- https://towardsdatascience.com/how-to-train-a-word2vec-model-from-scratch-with-gensim-c457d587e031

Understanding Doc2vec
- https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e
- https://towardsdatascience.com/detecting-document-similarity-with-doc2vec-f8289a9a7db7

## Tutorials
- Gensim, Doc2vec: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py

## dataset
- job posting dataset - https://data.world/promptcloud/indeed-usa-job-listing-data
- resumes corpus - https://github.com/florex/resume_corpus


## ToDos
1. baseline doc2vec model using cosine distance matching
1. organize resumes_corpus in to one json.gz file with {label: "label fileName.lbl", text: "resume text fileName.txt"}
1. cluster resumes and job descriptions by category 
1. improve resume matching using clustering and doc2vec model
1. Organize datasets
    - resume dataset
    - job description
    - use the categories from the jd dataset to be applied to the 


## Findings
- Baseline performance: 
    - Doc2Vec model trained using a single article from (https://www.analyticsvidhya.com/blog/2023/04/apoorvas-journey-of-challenges-and-growth-as-a-data-scientist/)
    - 
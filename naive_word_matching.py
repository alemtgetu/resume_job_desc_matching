# Python3 program for Naive Pattern
# Searching algorithm

from collections import Counter


def naive_matching(skills, jd):
    
    skills_lst = skills.split(',')
    
    counter   = Counter()
    N = len(jd)
    
    for skill in skills_lst:
        M = len(skill)
        # N = len(txt)
        for i in range(N - M + 1):
            j = 0
    
            # For current index i, check
            # for pattern match */
            while(j < M):
                if (jd[i + j] != skill[j]):
                    break
                j += 1
    
            if (j == M):
                print("Pattern found at index ", i)
                counter[skill]+=1
        
    print(counter)
 
# start here
if __name__ == '__main__':
    
    jd = "aws, python, leadership, machine learning"
    skills = "support vector machine, python, team leader"
    
    naive_matching(skills, jd)
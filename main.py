import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords 
from string import punctuation
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict
import pandas as pd 
from nltk.collocations import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request
import mysql.connector
import json


app = Flask(__name__)

#def combine_company_details():

#    return combined_column

def get_recommendation(top, df_all, scores):
    recommendation = pd.DataFrame(columns = ['id','name','categories','description'])
    count = 0
    for i in top:
        recommendation.at[count, 'id'] = df_all['project_id'][i]
        recommendation.at[count,'name'] = df_all['name'][i]
        recommendation.at[count,'categories'] = df_all['categories'][i]
        recommendation.at[count,'description'] = df_all['description'][i]
        #recommendation.at[count,'experience'] = df_all['experience'][i]
        #recommendation.at[count,'skill'] = df_all['Skill'][i]
        recommendation.at[count,'score'] = scores[count]
        count +=1
    return recommendation

def TFIDF(company_detail, candidate_detail):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    #fit and counting transforming the vector
    tfidf_job = tfidf_vectorizer.fit_transform(company_detail)

    tfidf_candidate = tfidf_vectorizer.transform(candidate_detail)

    cos_similarity_tfidf = map(lambda x: cosine_similarity(tfidf_candidate, x), tfidf_job)

    TFIDF_output = list(cos_similarity_tfidf)
    
    #top = sorted(range(len(TFIDF_output)), key=lambda i:TFIDF_output[i], reverse=True)[:100]
    #list_scores = [TFIDF_output[i][0][0] for i in top]
    #TF = get_recommendation(top, df, list_scores)

    return TFIDF_output

def count_vectorize(company_detail, candidate_detail):
    count_vectorizer = CountVectorizer()

    count_job = count_vectorizer.fit_transform(company_detail)

    count_candidate = count_vectorizer.transform(candidate_detail)

    cos_similarity_countV = map(lambda x:cosine_similarity(count_candidate,x), count_job)

    count_vectorize_output = list(cos_similarity_countV)

    #top = sorted(range(len(count_vectorize_output)), key=lambda i: output3[i], reverse=True)[:100]
    #list_scores = [count_vectorize_output[i][0][0] for i in top]
    #CV=get_recommendation(top, df, list_scores)
    return count_vectorize_output

def KNN(company_detail, candidate_detail):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    KNN = NearestNeighbors(n_neighbors=4)

    KNN.fit(tfidf_vectorizer.fit_transform(company_detail))

    NNs = KNN.kneighbors(tfidf_vectorizer.transform(candidate_detail))
    
    #top = NNs[1][0][1:]

    #index_score = NNs[0][0][1:]

    #knn_output = get_recommendation(top, top, index_score)

    return NNs

def merge_scaling(knn, tfidf, cv):

    merge1=knn[['id','name','score']].merge(tfidf[['id','score']], on='id')
    merge2=merge1.merge(cv[['id','score']], on='id')
    final = merge2.rename(columns={"score_x": "KNN", "score_y": "TF-IDF","score": "CV"})
    slr = MinMaxScaler()

    final[["KNN", "TF-IDF", 'CV']] = slr.fit_transform(final[["KNN", "TF-IDF", 'CV']])

    # Multiply by weights
    final['KNN'] = (1-final['KNN'])/3
    final['TF-IDF'] = final['TF-IDF']/3
    final['CV'] = final['CV']/3
    final['Final'] = final['KNN']+final['TF-IDF']+final['CV']
    final=final.sort_values(by="Final", ascending=False)

    return final

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    #connect to mysql
    id = request.json['input']
    try:
        connection = mysql.connector.connect(host='localhost',
                                            database='ppms-v6',
                                            user='root',
                                            password='')

        cursor = connection.cursor()
        #get all project detail
        sql_query = """SELECT project_id, name, categories, description FROM practicum_projects join academic_sessions on practicum_projects.academic_session_id = academic_sessions.academic_session_id where status='Open' and academic_sessions.active=1"""
        cursor.execute(sql_query)
        record = cursor.fetchall()

        #convert all project from sql query into dataframe
        sql_data = pd.DataFrame(record, columns=['project_id','name','categories','description'])
        sql_data2 = sql_data.copy()

        #get one project
        project = pd.DataFrame(record, columns=['project_id','name','categories','description'])
        project = sql_data.query('project_id=='+id)
        project['all'] = project['description'] + " " + project['name'] + " " + project['categories']

        sql_data2 = sql_data2.query('project_id!='+id)
        print(sql_data2)
        combined = pd.DataFrame(columns=['all'])
        combined['all'] = sql_data2['description'] + " " + sql_data2['name'] + " " + sql_data2['categories']

        #get recommendation using tdidf
        tfidf = TFIDF(combined['all'], project['all'].head(1))
        top_tfidf = sorted(range(len(tfidf)), key=lambda i:tfidf[i], reverse=True)[:100]
        list_scores_tfidf = [tfidf[i][0][0] for i in top_tfidf]
        TF = get_recommendation(top_tfidf, sql_data, list_scores_tfidf)

        #get recommendation using count vectorizer
        countV = count_vectorize(combined['all'], project['all'].head(1))
        top_countV = sorted(range(len(countV)), key=lambda i:countV[i], reverse=True)[:100]
        list_scores_countV= [countV[i][0][0] for i in top_countV]
        CV = get_recommendation(top_countV, sql_data, list_scores_countV)

        #get recommendation using KNN
        knn = KNN(combined['all'], project['all'].head(1))
        top_knn = knn[1][0][1:]
        index_score_knn = knn[0][0][1:]
        knn_result = get_recommendation(top_knn, sql_data, index_score_knn)

        #merge and scaling for all methods
        final_recommendation=merge_scaling(knn_result,TF,CV)
        final_recommendation = final_recommendation.query('id!='+id)
        json_final_recommendation= final_recommendation['id']
        print(json_final_recommendation)
        return json.dumps(json.loads(json_final_recommendation.to_json(orient="split", index=False)))
        
    except mysql.connector.Error as error:
        print("Failed to load data")

    finally:
        if connection.is_connected():
           cursor.close()
           connection.close()
           print("MySQL connection is closed")


@app.route('/')
def index():
    return "Testing recommendation engine"

if __name__ == '__main__':
    app.run()

    


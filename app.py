from flask import Flask,request,render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

df = pd.read_csv('model/movies.csv')
df['genre'] = df['genre'].str.replace(',',' ')
df['director']=df['director'].str.replace(' ','')
df['cast']=df['cast'].str.replace(' ','')
df['cast']=df['cast'].str.replace(',',' ')
df['title']=df['title'].str.lower()
df.sort_values(by=('year'),axis=0,inplace=True,ascending=False) 
df.reset_index(drop=True,inplace=True) 
df['year'] = df['year'].astype(str)
df['features']=df['year']+" "+df['genre']+" "+df['country']+" "+df['cast']
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['features'])
sim = cosine_similarity(count_matrix)

def recommender(name):
    name = name.lower()
    if name not in df['title'].unique():
        return ('Try Aboslute Spelling or it may not be in our database')
    else:
        i = df.loc[df['title']==name].index[0]
        mov_id = list(enumerate(sim[i]))
        mov_id = sorted(mov_id,key = lambda x:x[1],reverse=True)
        mov_id = mov_id[1:11]
        rec_mov = []
        for i in range(len(mov_id)):
            a = mov_id[i][0]
            rec_mov.append(df['title'][a])
        return rec_mov

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('main.html')

@app.route('/predict', methods=["GET","POST"])
def predict():
    movie_name = request.form.get('movie')
    predictions = recommender(movie_name)
    return render_template('trueReturn.html', predictions=predictions)
    
if __name__=='__main__':
    app.run(debug=True)

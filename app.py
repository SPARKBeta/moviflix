from flask import Flask,request,render_template
import pickle
import pandas as pd

app = Flask(__name__)
model_file = pickle.load(open('./model/model.pkl','rb'))
def recommender(name,model_file):
    df = pd.read_csv('model/movies.csv')
    df['title']=df['title'].str.lower()
    df.sort_values(by=('year'),axis=0,inplace=True,ascending=False) 
    df.reset_index(drop=True,inplace=True) 
    name = name.lower()
    if name not in df['title'].unique():
        return ('Try Aboslute Spelling or it may not be in our database')
    else:
        i = df.loc[df['title']==name].index[0]
        mov_id = list(enumerate(model_file[i]))
        mov_id = sorted(mov_id,key = lambda x:x[1],reverse=True)
        mov_id = mov_id[1:11]
        rec_mov = []
        for i in range(len(mov_id)):
            a = mov_id[i][0]
            rec_mov.append(df['title'][a])
        return rec_mov

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=["GET","POST"])
def predict():
    movie_name = request.form.get('movie')
    predictions = recommender(movie_name,model_file)
    return render_template('trueReturn.html', predictions=predictions)
    
if __name__=='__main__':
    app.run(debug=True)

# flask, skikit-learn, pandas, pickle-mixin
from flask import Flask, render_template, request
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from string import punctuation
app = Flask(__name__)

data = pd.read_csv(r"D:\Language\Machine Learning\6. Machine Learning\US Airline Project\training_twitter_x_y_train.csv")
v = pickle.load(open(r"D:\Language\Machine Learning\6. Machine Learning\US Airline Project\v.pkl", "rb"))
clf = pickle.load(open(r"D:\Language\Machine Learning\6. Machine Learning\US Airline Project\SVM.pkl", "rb"))
@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods = ["POST"]) 
def predict():
    Airline  = request.form.get('Airline')
    Name = request.form.get('Name')
    negativegoldreason = request.form.get('negativegoldreason')
    text = request.form.get('text')
    print(Airline, Name, negativegoldreason, text)
    input = pd.DataFrame([[Airline, negativegoldreason, text]],columns = ['airline', 'negativereason_gold', 'text'])
    stops = stopwords.words('english')
    stops += list(punctuation)
    stops += ['flight', 'airline', 'flights', 'AA']
    abbreviations ={'ppl' : 'people','cust':'customer','serv':'service','mins':'minutes','hrs':'hours','svc': 'service','u':'you','pls':'please' }
    input_index = input[~input.negativereason_gold.isna()].index
    tweet = input.text
    tweet = str(tweet) 
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet) #remove links
    tweet = re.sub('@[^\s]+','',tweet) #remove usernames
    tweet = re.sub('[\s]+', ' ', tweet) #remove additional whitespaces
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) #replace #word with word
    tweet = tweet.strip('\'"') #trim tweet
    words = []
    for word in tweet.split(): 
#         if not hasNumbers(word):
        if word.lower() not in stops:
            if word in list(abbreviations.keys()):
                words.append(abbreviations[word])
            else:
                words.append(word.lower())
    tweet = " ".join(words)
    tweet = " %s %s" % (tweet, input.airline)
    input.text = tweet
    if input.negativereason_gold.any():
        input.text = " %s %s" % (input.text, input.negativereason_gold)
    input.text = str(input.text).encode('ascii', 'ignore').decode('ascii')     
    words = str(input.text).split()
    new_words = []
    for word in words:
        if not any(char.isdigit() for char in word):
            new_words.append(word)
    input.text = " ".join(new_words)
    input.text = v.transform(input.text)
    prediction = clf.predict(input.text)
    print(prediction)
    return str(prediction)
if __name__== "__main__":
    app.run(debug = True, port = 5001)
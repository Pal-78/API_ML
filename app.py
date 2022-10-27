#!/usr/bin/env python
# coding: utf-8

# In[19]:


import nltk
#import contractions
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, words
import string
import re
import flask
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


# In[20]:


app = Flask(__name__)


# In[21]:


# initialisation
tk = TweetTokenizer(reduce_len=True)
stop_words_e = set(stopwords.words('english'))
words_e = set(words.words())
lemmatizer = WordNetLemmatizer()
# regex compilation to be faster
# remove URLs
re_url = re.compile(r'https?://[^ ]+')
re_url1 = re.compile(r'www\.[^ ]+')
# remove usernames
re_user = re.compile(r'@[^ ]+')
# remove all but lowercase letters and space
re_l = re.compile(r'[^a-z ]+')


# In[22]:


# load model
model = load_model('model.h5')
# load tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)


# In[23]:


def pre_process_lem(text):
    """
    lowercase
    translate emojis :)
    remove URLs
    remove usernames references
    expanding contractions
    translate abbreviation
    take out punctuation, including hashtags, get rid of numbers
    transform into list of tokens, tweeter style
    take out stopwords
    lemmatize
    keep english words only
    build up sentences with remaining words
    """

    # Defining dictionary containing emojis with their meanings.
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
# defining dictionay containing all contractions to translate into real english words
    abbreviations = {
    "2day" : "today",
    "2morow" : "tomorrow",
    "4ao" : "for adults only",
    "4got" : "forget",
    "4gotten" : "forget",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk",
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart",
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrs" : "hours",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mins" : "minutes",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "pls" : "please",
    "plz" : "please",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens",
    "r" : "are",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet",
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "secs" : "seconds",
    "sfw" : "safe for work",
     "sk8" : "skate",
    "smh" : "shake my head",
    "some1" : "someone",
    "sq" : "square",
    "srsly" : "seriously",
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "yrs" : "years",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
    }

    # lowercase
    text = text.lower()
    # translate emojis :)
    for emoji in emojis.keys():
        text = text.replace(emoji, emojis[emoji])
    # remove URLs
    text = re_url.sub(r'', text)
    text = re_url1.sub(r'', text)
    # remove usernames references
    text = re_user.sub(r'', text)
    # expand contractions
    expand_words = []
    for wd in text.split():
        #expand_words.append(contractions.fix(wd))
        expand_words.append(wd)
    text = ' '.join(expand_words)
    # translate abbreviation
    translate_words = []
    for wd in text.split():
        if wd in abbreviations.keys():
            wd = abbreviations[wd]
        translate_words.append(wd)
    text = ' '.join(translate_words)
    # take out punctuation, including hashtags, get rid of numbers
    text = re_l.sub(r'', text)
    # transform into list of tokens, tweeter style
    tokenize_sentence = tk.tokenize(text)
    # take out stopwords
    words_w_stopwords = [i for i in tokenize_sentence if i not in stop_words_e]
    #     lemmatize
    words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords)
    # build up sentences with remaining words
    sentence_clean = " ".join(w for w in words_lemmatize if w in words_e)

    return sentence_clean


# In[24]:


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Tweet Sentiment Analyzer</h1><p>Add your tweet on /sentiment/ or POST Jason data on /api</p>"




# In[26]:


# curl -X POST -H"Content-Type:application/json" -d@tweet.json http://url/api
@app.route('/api', methods=['POST','GET'])
def process_api():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json_data = request.json
    else:
        return 'Content-Type not supported!'
    results = {}
    for keys in json_data:
        tweet_p = pre_process_lem(json_data[keys])
        tweet_ps = pad_sequences(tokenizer.texts_to_sequences([tweet_p]),30)
        score = round(model.predict(tweet_ps)[0][0],2)
        label = 'POSITIVE'
        if score < 0.5:
            label = 'NEGATIVE'
        results[keys] = [label,str(score)]
    return jsonify(results)


# In[27]:

#if __name__ == '__main__':
#    app.run(debug=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:

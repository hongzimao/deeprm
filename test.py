import pickle

with open('data/pg_re_100.pkl', 'rb') as f:
    data = pickle.load(f)
    print data

f.close()

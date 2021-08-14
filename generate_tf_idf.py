'''
This script creates the tf-idf scores for the whole dataset. Each row is a combination of Order Title and Line Description for an order.
'''


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import config as config
import sys
import pickle


df = pd.read_csv(config.datasets_dir+config.clean_csv_name)

tfidf_vector = TfidfVectorizer(max_df = 0.98, min_df = 0.02, ngram_range=(1,6))
transformed = tfidf_vector.fit_transform(df['text_fields'])

df_tfidf = pd.DataFrame(transformed.toarray(), columns=tfidf_vector.get_feature_names())
important_words = []

def get_important_words(row):
	header = df_tfidf.columns
	t_dict = dict(zip(header,row))
	sorted_dict = dict(sorted(t_dict.items(), key=lambda item: item[1], reverse=True))
	#important_words.append(('    '.join(list(sorted_dict.keys())[:config.num_of_important_key_words])))
	important_words.append(('_'.join(list(sorted_dict.keys())[:5])))

df_tfidf.apply(get_important_words,axis=1)
df_tfidf["important_words"] = important_words

df_tfidf.to_csv(config.datasets_dir+config.tfidf_file_name)

clean_df = pd.concat([df,df_tfidf],axis=1)

clean_df.to_csv(config.datasets_dir+config.clean_tfidf_file_name,index=False)

df_tfidf["text_fields"] = df['text_fields']
df_tfidf["ASSET_CLASS"] = df["ASSET_CLASS"]
df_tfidf[["text_fields","ASSET_CLASS","important_words"]].to_csv(config.datasets_dir+"test_5.csv",index=False)









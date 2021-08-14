'''
This script performs the pre-processing on the raw dataset and shapes it as required by the shallow learning models. They will be trained on this pre-processed dataset.

'''

import boto3
import pandas as pd
#from sagemaker import get_execution_role
#from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
import config as config
from nltk.stem import WordNetLemmatizer
#from pattern.en import lemma
from nltk.corpus import stopwords
pd.set_option('display.max_colwidth', None)
from nltk.stem.snowball import SnowballStemmer
snow_stemmer = SnowballStemmer(language='english')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def connect():
	#role = get_execution_role()
	bucket=config.bucket
	data_key = config.data_key
	data_location = 's3://{}/{}'.format(bucket, data_key)
	return data_location

def pre_process(x):
    
    #x = re.sub('[^a-zA-Z0-9 \n\.]', ' ', x)
    x = re.sub('[^a-zA-Z]', ' ', x)
    x = x.split()
    for index,word in enumerate(x):
        if word in config.replacement_dict.keys():
            x[index] =  config.replacement_dict[word]
            
        x[index] = lemmatizer.lemmatize(x[index].lower())
      
    #x = [word for word in x if not word in stop_words and len(word)>2]  #uncommen thisline for getting full words and comment the below line
    x = [snow_stemmer.stem(word) for word in x if not word in stop_words and len(word)>2]
    x = ' '.join(x)
    x = re.sub('[^a-zA-Z]', ' ', x)
    x = re.sub(' +',' ',x)
    return(x)

def main():

	data_location = connect()
	df = pd.read_csv(data_location)
	#replace NaNs with empty string
	#print(df.isnull().any()) #NaN values present in FUND_SUBOBJCLASS
	df = df.replace(np.nan, '', regex=True)
	#print(df.isnull().any()) #NaN values replaced with white spaces

	df['SUB_OBJ_DESCR'] = df['SUB_OBJ_DESCR'].apply(pre_process)
	df['OBJ_CODE'] = df['OBJ_CODE'].apply(pre_process)
	df['ORDER_TITLE'] = df['ORDER_TITLE'].apply(pre_process)
	df["LINE_DESCRIPTION"] = df["LINE_DESCRIPTION"].apply(pre_process)
	df["VENDOR_COUNTRY"] = df["VENDOR_COUNTRY"].str.lower()
	df["VENDOR_NAME"] = df["VENDOR_NAME"].str.lower()

	#df["text_fields"] = df['SUB_OBJ_DESCR']+' '+df['OBJ_CODE']+' '+df['ORDER_TITLE']+' '+df["LINE_DESCRIPTION"]+' '+df["VENDOR_NAME"]+' '+df["VENDOR_COUNTRY"]
	df["text_fields"] = df['ORDER_TITLE']+' '+df["LINE_DESCRIPTION"]
	
	print(df["text_fields"].head(100))

	df.to_csv(config.datasets_dir+config.clean_csv_name,index=False)

if __name__ == "__main__":
	main()







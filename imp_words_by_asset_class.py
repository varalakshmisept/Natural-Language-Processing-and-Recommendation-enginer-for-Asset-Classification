import pandas as pd
import config as config
import sys

'''
This program groups the important topics identified and groups them by asset classes.
'''

def set_of_important_words_by_class():
	
	dict_of_words = {}
	df = pd.read_csv(config.datasets_dir+config.clean_tfidf_file_name)
	
	
	#print('---'.join(df.columns))
	groups = df.groupby(by=["ASSET_CLASS"])
	#print(groups.groups.keys())

	for name,group in groups:
		replaced_str = group["important_words"].str.replace(' ','_')
		dict_of_words[name] = ' '.join(list(set(replaced_str.str.cat(sep='_').split('_'))))

	pd.DataFrame(dict_of_words.items(), columns=['ASSET_CLASS', 'important_words']).to_csv(config.datasets_dir+config.important_word_by_class_csv_file_name,index=False)


if __name__ == "__main__":
	set_of_important_words_by_class()


		




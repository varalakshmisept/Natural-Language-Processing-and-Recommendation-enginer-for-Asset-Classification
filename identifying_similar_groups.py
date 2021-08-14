'''
This script uses a custom logic to group the similar asset classes over the tf-idf values. This program needs to be run multiple times by commenting and uncommenting various sections. They are mentioned in the comments run1 and run2.

'''

import json
from sklearn.metrics.pairwise import linear_kernel
import config as config
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import sys
import numpy as np
from collections import Counter
import csv
import dask.dataframe as dd 
import os
import datetime
from sklearn import preprocessing

#high_list = []


def filter_vaues(x):
	dict_ = zip(cols,x)
	dict_ = {k:v for k,v in dict(dict_).items() if v>=0.7}
	categories = {'category_1':[],'category_2':[],'category_3':[]}

	for key,value in dict_.items():
		value = round(value,2)
		if 0.9 <= value <=1:
			if len(categories['category_1'])==0:
				categories['category_1'] = [asset_classes[key]]
			else:
				categories['category_1'].append(asset_classes[key])
		elif 0.8 <= value < 0.9:
			if len(categories['category_2'])==0:
				categories['category_2'] = [asset_classes[key]]	
			else:
				categories['category_2'].append(asset_classes[key])
		elif 0.7 <= value < 0.8:
			if len(categories['category_3'])==0:
				categories['category_3'] = [asset_classes[key]]
		else:
			categories['category_3'].append(asset_classes[key])

	high_list.append(categories)

	
def counter_func(x):
	counter_dict = Counter(x.split()).most_common(5)
	return(dict(counter_dict))



def concat_similarities(df_func,cat_col_name,percentage_similarity_col_names):
        df_func["ASSET_CLASS"] = df_func.index
        #df_func["ASSET_CLASS"] = df_func["ASSET_CLASS"].to_string()
        print(df_func)
        print(df_func.dtypes)
        for index, row in  df_func.iterrows():
                col_index = 0
                for key,value in row[cat_col_name].items():
                        df_func.loc[index,col_names_cat[col_index]] = key+":"+str(value)
                        if value>0: 
                                #index_of_the_matching_class = df_func[df_func["ASSET_CLASS"]==key].index.tolist()[0]
                                #print(index_of_the_matching_class)
                                try:
                                    #print("*"*100)
                                    #print(df_func[df_func["ASSET_CLASS"]==key].index.tolist())
                                    index_of_the_matching_class = df_func[df_func["ASSET_CLASS"]==key].index.tolist()[0]
                                    #print(index_of_the_matching_class)
                                    #print(df_func.loc[index_of_the_matching_class,"Total"])
                                    df_func.loc[index,percentage_similarity_col_names[col_index]] = float(value/df_func.loc[index_of_the_matching_class,"Total"])
                                except:
                                    df_func.loc[index,percentage_similarity_col_names[col_index]] = 0
	
                        col_index=col_index+1



#remove wherever you see 50 or head(50)

def dump_final_optimized_classes_csv():

    
    original_dataset = pd.read_csv(config.datasets_dir+config.clean_tfidf_file_name)  #!!!!!!!!!!!!!!!! - remove here ------!!!!!!

    match_df_orig = pd.read_csv(config.datasets_dir+config.final_matching_df)
    #match_df_orig = match_df_orig[match_df_orig['actual_class'] != match_df_orig['matching_class']]
    match_df_orig.drop_duplicates(keep=False,inplace=True)
    match_df_orig.reset_index(drop=True,inplace=True)
    match_df_orig = match_df_orig.applymap(str)

    le = preprocessing.LabelEncoder()

    match_df = pd.DataFrame()
    le.fit(original_dataset["ASSET_CLASS"].unique())
    #print(le.classes_)

    match_df["actual_class"] = le.transform(match_df_orig["actual_class"])
    match_df["matching_class"] = le.transform(match_df_orig["matching_class"])
    match_df = match_df.applymap(str)

    #print(match_df)

    #print(original_dataset["ASSET_CLASS"].unique())
    original_dataset['ASSET_CLASS_OLD'] = original_dataset['ASSET_CLASS'] 
    original_dataset['ASSET_CLASS'] = le.transform(original_dataset['ASSET_CLASS'])
    original_dataset['ASSET_CLASS'] = original_dataset['ASSET_CLASS'].apply(str)


    match_df = match_df[match_df['actual_class'] != match_df['matching_class']]
    match_df_dict = dict(zip(match_df['actual_class'],match_df['matching_class']))

    k_v_exchanged = {}
    for key, value in match_df_dict.items():
        if value not in k_v_exchanged:
            k_v_exchanged[value] = [key]
        else:
            k_v_exchanged[value].append(key)

    rightside_combinations = {k:v for (k,v) in k_v_exchanged.items() if len(v) > 1}
    #print(rightside_combinations)

    substitute_classes = [key+" "+" ".join(values) for key,values in rightside_combinations.items()]
    #print(substitute_classes)


    for class_ in substitute_classes:
        classes = class_.split(" ")
        for every_class in classes:
            original_dataset['ASSET_CLASS'] = original_dataset['ASSET_CLASS'].replace({every_class:class_})
            match_df = match_df.replace({every_class:class_})

    match_df = match_df.drop_duplicates(keep=False)
    match_df.reset_index(drop=True,inplace=True)

    print(match_df)
    #------------------------------------------------------------------------------------------------------- 
    match_df = match_df[match_df['actual_class'] != match_df['matching_class']]
    match_df_dict = dict(zip(match_df['actual_class'],match_df['matching_class']))

    k_v_exchanged = {}
    for key, value in match_df_dict.items():
        if value not in k_v_exchanged:
            k_v_exchanged[value] = [key]
        else:
            k_v_exchanged[value].append(key)

    rightside_combinations = {k:v for (k,v) in k_v_exchanged.items() if len(v) > 1}
    #print(rightside_combinations)

    substitute_classes = [key+" "+" ".join(values) for key,values in rightside_combinations.items()]
    #print(substitute_classes)


    for class_ in substitute_classes:
        classes = class_.split(" ")
        for every_class in classes:
            original_dataset['ASSET_CLASS'] = original_dataset['ASSET_CLASS'].replace({every_class:class_})
            match_df = match_df.replace({every_class:class_})

    match_df = match_df.drop_duplicates(keep=False)
    match_df.reset_index(drop=True,inplace=True)

    print(match_df)
    #-----------------------------------------------------------------------------------------------------


    for i in range(0,len(match_df)):
        uniq_asset_classes = original_dataset["ASSET_CLASS"].unique()
        print(uniq_asset_classes)

        index = [idx for idx, s in enumerate(uniq_asset_classes) if  match_df.loc[i,"actual_class"] in s][0]
        print("Merge \t"+match_df.loc[i,"matching_class"]+"\t with \t"+uniq_asset_classes[index])

        replacement_string = ' '.join(set((match_df.loc[i,"matching_class"]+' '+uniq_asset_classes[index]).split(" ")))
        original_dataset['ASSET_CLASS'] = original_dataset['ASSET_CLASS'].replace({match_df.loc[i,"matching_class"]:replacement_string, match_df.loc[i,"actual_class"]:replacement_string})

    print(len(original_dataset['ASSET_CLASS'].unique()))

    #original_dataset.to_csv(config.datasets_dir+config.optimized_dataset,index=False)


    print(le.classes_)

    
    global optimized_asset_classes
    optimized_asset_classes = []


    def get_class_names(x):
        #print("*"*100)
        #print(le.inverse_transform(list(map(int, x.split(' ')))))
        optimized_class_str = ' '.join(list(le.inverse_transform(list(map(int, x.split(' '))))))
        optimized_asset_classes.append(optimized_class_str)

    original_dataset['ASSET_CLASS'].apply(get_class_names)
    original_dataset['OPTIMIZED_CLASSES'] = optimized_asset_classes
    print(original_dataset['OPTIMIZED_CLASSES'])
    print(original_dataset['OPTIMIZED_CLASSES'].unique())
    print(len(original_dataset['OPTIMIZED_CLASSES'].unique()))

    #writing asset classes to a CSV
    pd.Series(original_dataset['OPTIMIZED_CLASSES'].unique()).to_csv(config.results_dir+config.optimized_asset_classes,index=False)
    print("Unique Asset classes - optimized and written to CSV")

    

        



def generate_category_dfs(): 
        

        high_df = pd.read_csv(config.datasets_dir+config.categories_and_classes,header=None)

        #REMOVE THISSSSSSSS
        #high_df = high_df.head(50)          #!!!!!!!!!!!!!!!!!!!!!!!!--------- remove heree--------------------------!!!!!!!!!#

        high_df.columns = ['category_1','category_2','category_3','ASSET_CLASS']
        high_df['ASSET_CLASS'] = high_df['ASSET_CLASS'].apply(str)

        #print(high_df['ASSET_CLASS'].unique())

        print(high_df)
        print("*"*50)
        
        global cat_1_df
        global cat_2_df
        global cat_3_df
	
        
        cat_1_df = pd.DataFrame()
        cat_2_df = pd.DataFrame()
        cat_3_df = pd.DataFrame()


	#print(high_df.groupby('ASSET_CLASS'))
        
        total_rows_in_class = []
        
        high_df = high_df.dropna(subset=['category_1'])

        high_df_groups = high_df.groupby('ASSET_CLASS')

        for key, item in high_df_groups:
            print(key)
            print(len(item))
            total_rows_in_class.append(len(item))
        

        print("#"*100)

        cat_1_df = high_df_groups.agg({'category_1':' '.join})
        #cat_2_df = high_df.groupby('ASSET_CLASS').agg({'category_2':' '.join})
        #cat_3_df = high_df.groupby('ASSET_CLASS').agg({'category_3':' '.join})
       
        #print(cat_1_df)

        print("Aggregation for Asset Classes done ... ... ...")
        
        cat_1_df['Total'] = total_rows_in_class
        #cat_2_df['Total'] = total_rows_in_class
        #cat_3_df['Total'] = total_rows_in_class
	
        cat_1_df["category_1"] = cat_1_df["category_1"].apply(counter_func)
        #cat_2_df["category_2"] = cat_2_df["category_2"].apply(counter_func)
        #cat_3_df["category_3"] = cat_3_df["category_3"].apply(counter_func)
       
         
        global col_names_cat

        col_names_cat=['1_most_similar','2_most_similar','3_most_similar','4_most_similar','5_most_similar','percentage_similarity_1','percentage_similarity_2','percentage_similarity_3','percentage_similarity_4','percentage_similarity_5']	
        percentage_similarity_col_names = ['percentage_similarity_1','percentage_similarity_2','percentage_similarity_3','percentage_similarity_4','percentage_similarity_5']
        
        #dfs_list = [cat_1_df,cat_2_df,cat_3_df]
        dfs_list = [cat_1_df]
	
        for every_df in dfs_list:
            for col_name in col_names_cat:
                every_df[col_name] = ""
       
        concat_similarities(cat_1_df,"category_1",percentage_similarity_col_names)
        #concat_similarities(cat_2_df,"category_2",percentage_similarity_col_names)
        #concat_similarities(cat_3_df,"category_3",percentage_similarity_col_names)
        
        cat_1_df["category_1"] = cat_1_df["category_1"].astype(str)
        #cat_2_df["category_2"] = cat_2_df["category_2"].astype(str)
        #cat_3_df["category_3"] = cat_3_df["category_3"].astype(str)

        print(cat_1_df)
	#print(cat_2_df)
	#print(cat_3_df)

        cat_1_df = cat_1_df[cat_1_df["Total"]<=500]

        global actual_class
        global matching_class

        actual_class=[]
        matching_class=[]

        def finding_max_match_class(x):
            percentage_similarity_list = [x['percentage_similarity_1'],x['percentage_similarity_2'],x['percentage_similarity_3'],x['percentage_similarity_4'],x['percentage_similarity_5']]
            #print(percentage_similarity_list)
            percentage_similarity_list = [el if isinstance(el, float) else 0 for el in percentage_similarity_list]
            max_match_index = percentage_similarity_list.index(max(percentage_similarity_list))
            #print(max_match_index)

            if(max(percentage_similarity_list) != 0):
                most_similar_list = [x['1_most_similar'],x['2_most_similar'],x['3_most_similar'],x['4_most_similar'],x['5_most_similar']]
                max_match_class = most_similar_list[max_match_index].split(":")[0]

                actual_class.append(x["ASSET_CLASS"])
                matching_class.append(max_match_class)



        cat_1_df.apply(finding_max_match_class,axis=1)    

        #print(cat_1_df)
        #print(cat_1_df.iloc[0,:])
        #print(actual_class)
        #print(matching_class)

        final_matching_df = pd.DataFrame()
        final_matching_df["actual_class"] = actual_class
        final_matching_df["matching_class"] = matching_class 


        final_matching_df.to_csv(config.datasets_dir+config.final_matching_df,index=False)
        

        dump_final_optimized_classes_csv()

        sys.exit(0)

        path = config.similar_classes_dir
        
        if os.path.exists(path+config.category_1_similarity_file_name):
            cat_1_df.to_csv(config.similar_classes_dir+config.category_1_similarity_file_name,mode ='a',index=False)
        else:
            cat_1_df.to_csv(config.similar_classes_dir+config.category_1_similarity_file_name,index=False)
        '''
        if os.path.exists(path+config.category_2_similarity_file_name):
            cat_2_df.to_csv(config.similar_classes_dir+config.category_2_similarity_file_name,mode ='a',index=False)
        else:
            cat_2_df.to_csv(config.similar_classes_dir+config.category_2_similarity_file_name,index=False)

        if os.path.exists(path+config.category_3_similarity_file_name):
            cat_3_df.to_csv(config.similar_classes_dir+config.category_3_similarity_file_name,mode ='a',index=False)
        else:
            cat_3_df.to_csv(config.similar_classes_dir+config.category_3_similarity_file_name,index=False)
        '''

        print("File written ... ... ...")


def calculations_on_similarity_df (similarity_df,chunk_size_start,chunk_size_end,start_time):
        print("read the csv")
        similarity_df = similarity_df.fillna(0)

        global cols
        cols = list(range(0,len(df)))
        similarity_df.columns = cols

        print("reading csv")
        similarity_df = similarity_df.astype(float)
        similarity_df.columns = similarity_df.columns.astype(int)
        similarity_df.values[[np.arange(similarity_df.shape[0])]*2] = 0
	#print(similarity_df)
	
	
        similarity_df.apply(filter_vaues,axis=1)
	#print(high_list)
        
        high_df = pd.DataFrame(high_list)
        #print(high_df)
 
        
        for col in list(high_df.columns):
            high_df[col] = high_df[col].str.join(' ')

	#print(high_df)	
        print("chunk_size_start",chunk_size_start)
        print("chunk_size_end",chunk_size_end)
        #print("array",asset_classes[chunk_size_start:chunk_size_end])
        #print(high_df)
        high_df["ASSET_CLASS"] =  asset_classes[chunk_size_start:chunk_size_end] #remove :6 and also remove if i==5
	#high_df["ASSET_CLASS"] =  asset_classes[:51] #remove :6 and also remove if i==5
        
        high_df.to_csv(config.datasets_dir+config.categories_and_classes,mode='a',index=False, header=None)
        end_time = datetime.datetime.now()
        print("end_time",end_time)
        tot_time = end_time - start_time
        print("tot_time",tot_time)
        




if __name__ == "__main__":


        # ---------------------------------------------------------------------------------------------------------------------------------------
        
        #start of run 1
        print("Execution started")
        df = pd.read_csv(config.datasets_dir+config.clean_tfidf_file_name)
        global asset_classes
        asset_classes  = df.iloc[:,11].to_list()
        df = df.iloc[:,14:-1]
        print("DF loaded")

        with open(config.datasets_dir+config.cosine_similarities_csv,"w") as file:
        
            writer = csv.writer(file, delimiter=',')
            writer.writerow(list(range(0,len(df))))
            print("Opened csv file... .. ..")
            for i in range(0,len(df)):
                print(i)
                similarities = cosine_similarity(df,df.iloc[i:i+1,:]).T
                similarities = similarities[similarities>=0.7]
                writer.writerow(list(similarities))
                #if i==50:
                #break
        print("calculated similarities ... ... ...")
        #end of run 1

        # ---------------------------------------------------------------------------------------------------------------------------------------

        #start of run 2

        print("Trying to read csv")  
        #similarity_df = pd.read_csv(config.datasets_dir+config.cosine_similarities_csv,low_memory=False,index_col = False,encoding='utf-8')
        chunksize = 1000
        print("chunksize",chunksize)
        chunk_size_end=chunksize
        chunk_size_start=0
        
        start_time = datetime.datetime.now()
        print("start_time",start_time)

        with pd.read_csv(config.datasets_dir+config.cosine_similarities_csv,index_col=False, chunksize=chunksize) as reader:

            global high_list 
            high_list = []

            print("inside with")
            for chunk in reader:
                #print("*"*100)
                #print(chunk)
                #print("*",100)
                chunk = chunk.round(decimals=2)
                print("*"*100)
                calculations_on_similarity_df(chunk,chunk_size_start,chunk_size_end,start_time)
                chunk_size_start = chunk_size_end
                chunk_size_end = chunk_size_end+chunksize
                high_list = []

        #end of run 2

        generate_category_dfs()


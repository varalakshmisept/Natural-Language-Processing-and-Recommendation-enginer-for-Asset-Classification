import pandas as pd
from predict_lstm import pre_process 
import config

'''
The script takes the values from the UI and inserts it into a new dataset that will further be used for continuous training.
'''

def pre_process_and_append_to_dataset(order_title,line_description,asset_class):

    df = pd.read_csv(config.datasets_dir+config.final_preprocessed)
    order_tile_pre_processed = pre_process(order_title,'')
    line_description_pre_processed = pre_process(line_description,'')
    position = len(df)+1

    df.loc[position,"ORDER_TITLE"] = order_tile_pre_processed
    df.loc[position,"LINE_DESCRIPTION"] = line_description_pre_processed
    df.loc[position,"SPELL_CORRECTED"] = order_tile_pre_processed+" "+line_description_pre_processed
    df.loc[position,"ASSET_CLASS"] = asset_class
    

    df.to_csv(config.deep_learning_dataset_for_retraining,index=False)
    return("successfully added the ORDER TITLE and LINE DESCRIPTION to the dataset")



if __name__=="__main__":
    pre_process_and_append_to_dataset('transformers% warehouse#123 location$ fap9989','transformers outlet made pastic case carrying transformer','39300')


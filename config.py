import os

base_dir = '/'.join(os.getcwd().split("/")[:-1])
bucket="daen690-centurions-data"
data_key = 'GMU_Procurement_AssetClass_Data.csv'
replacement_dict = {"SUPPL":"SUPPLIES","A C":"AC","a c":"AC","I PHONES":"iPhone","W H":"WH","AGYSERV":"agency service","ADVERTIS":"advertisement",
"BLDG":"building",
"CAPITLIZED":"capitalized","COMMUNCAT":"communication","COMM":"communication","CONTR":"contractor","CONT":"contract","COMM":"commodities",
"DEVEL":"development",
"EQUI":"equipment","EQUIP":"equip","EXPE":"expense","EX":"expense","EQUIPM":"equipment","EXPEN":"expense","ENGR":"engineer",
"FURN":"furniture",
"GOVT":"government","GENERATO":"generator","GARB":"garbage","GD":"guard",
"HSEHLD":"household",
"INST":"installation","INSURAN":"insurance","IMMUNIZN":"immunization",
"MISC":"miscellaneous","MAINT":"maintenance","MINIC":"minicomputer","MAINTENA":"maintenance",
"NETWORKI":"networking","NON-EXPD":"non expenditure",
"OTHR":"other","OPER":"operator","OTH":"other","OBLIG":"obligation",
"PROP":"property","PERSO":"person","PREP":"prepare","PERS":"personal",
"REPR":"repair","RES":"residence","REPRODUC":"reproduction",
"STD":"std","SER":"service","SUPPLS":"supplies","STRUCTU":"structure","SERV":"service","SOFTWAR":"software","SUPPLIE":"supplies","SYST":"system","SECUR":"security","SOFTW":"software",
"TEL":"telephone","TRAINI":"training","TRSH":"trash","TRANS":"transport","TRANSP":"transport"}

scripts_dir = base_dir+"/scripts/"
utils_dir = "/home/ubuntu/asset_classification/utils/"
plots_dir = "/home/ubuntu/asset_classification/plots/"
results_dir = "/home/ubuntu/asset_classification/results/"
datasets_dir = "/home/ubuntu/asset_classification/datasets/"
models_dir = "/home/ubuntu/asset_classification/models/"

tf_idf_vector_name = "tf_idf_vector.pkl"

# Model Data1
mlp_model_data1 = base_dir +"/models/"+"mlp_models_data1.hdf5"

similar_classes_dir = datasets_dir+"similar_classes/"
rf_results_data3 = results_dir+"rf_results_data3.csv"
dt_results_data3 = results_dir+"dt_results_data3.csv"
knn_results_data3 = results_dir+"knn_results_data3.csv"
nb_results_data3 = results_dir+"nb_results_data3.csv"
# Models
rf_model_data3 = base_dir+"/models/"+"rf_model_data3.sav"
dt_model_data3 = base_dir+"/models/"+"dt_model_data3.sav"
knn_model_data3 = base_dir+"/models/"+"knn_models_data3.sav"
nb_model_data3 = base_dir+"/models/"+"nb_models_data3.sav"


rf_model_data1 = "/home/ubuntu/asset_classification/"+"/models/"+"rf_model_data1.sav"
dt_model_data1 = "/home/ubuntu/asset_classification/"+"/models/"+"dt_model_data1.sav"
knn_model_data1 = "/home/ubuntu/asset_classification/"+"/models/"+"knn_model_data1.sav"
nb_model_data1 = "/home/ubuntu/asset_classification/"+"/models/"+"nb_model_data1.sav"

rf_model_data2 = "/home/ubuntu/asset_classification/"+"/models/"+"rf_model_data2.sav"
dt_model_data2 = "/home/ubuntu/asset_classification/"+"/models/"+"dt_model_data2.sav"
knn_model_data2 = "/home/ubuntu/asset_classification/"+"/models/"+"knn_model_data2.sav"
nb_model_data2 = "/home/ubuntu/asset_classification/"+"/models/"+"nb_model_data2.sav"


mlp_model_data3 = base_dir +"/models/" +"mp_models_data3.hdf5"
clean_csv_name = datasets_dir + "cleaned_dataset.csv"
visualizations = base_dir+"/visualizations"

n_grams = 3
tfidf_file_name = "tfidf.csv"
num_of_important_key_words = 5
clean_tfidf_file_name = "cleaned_tfidf.csv"

important_word_by_class_csv_file_name = "important_words_by_class.csv"

categories_and_classes = "categories_and_classes.csv"

category_1_similarity_file_name = "category_1_similarity.csv"
category_2_similarity_file_name = "category_2_similarity.csv"
category_3_similarity_file_name = "category_3_similarity.csv"

cosine_similarities_csv = "cosine_similarities.csv"
final_matching_df = "final_matching_df.csv"

preprocessed_filename_deep_learning = "cleaned_data_for_deep_learning.csv"
final_preprocessed = "final_preprocessed.csv"
optimized_asset_classes = "optimized_asset_classes_groups.csv"

glove_embeddings_dict = utils_dir+"glove_embeddings_dict.pkl"
word_index_lstm = utils_dir+"word_index_lstm.pkl"
max_length_sentence_lstm = "/home/ubuntu/asset_classification/utils/"+"max_length_sentence_lstm.pkl"
tokenizer_lstm = "/home/ubuntu/asset_classification/utils/"+"tokenizer_lstm.pkl"
code_asset_class_mapping_dict = utils_dir+"code_asset_class_mapping_dict.pkl"
glove_embeddings_dict_keys = "/home/ubuntu/asset_classification/utils/"+"glove_embeddings_dict_keys.pkl"


optimized_dataset = "optimized_dataset.csv"

glove_txt_300d = "glove.840B.300d.txt"
correctspelling = "correct_spelling_dict.txt"
correctspellingnew = "correct_spelling_dict_new.txt"
cnn_dataset1 = base_dir+"/models/"+"cnn_dataset1.h5"
cnn_dataset3=base_dir+"/models/"+"cnn_datatset3.h5"
cnn_dataset2 = base_dir+"/models/"+"cnn_dataset2.h5"
cnn_prepocessed_dataset1 = base_dir+"/models/"+"cnn_preprocessed_dataset1.h5"
cnn_prepocessed_dataset2 = base_dir+"/models/"+"cnn_preprocessed_dataset2.h5"
cnn_dataset3_prepocessed = base_dir+"/models/"+"cnn_dataset3_preprocessed.h5"

lstm_prepocessed_dataset1 = base_dir+"/models/"+"lstm_preprocessed_dataset1.h5"
bilstm_prepocessed_dataset1 = base_dir+"/models/"+"bilstm_preprocessed_dataset1.h5"

dict_asset_class = datasets_dir + "dict_asset_class.pickle"
X_test_data1 = "/home/ubuntu/asset_classification/datasets/"+"X_test_data1.csv"
X_train_data1 = "/home/ubuntu/asset_classification/datasets/"+"X_train_data1.csv"
Y_train_data1 = "/home/ubuntu/asset_classification/datasets/"+"Y_train_data1.csv"
Y_test_data1 = "/home/ubuntu/asset_classification/datasets/"+"Y_test_data1.csv"

performance = "/home/ubuntu/asset_classification/datasets/" + "performance_metrics.csv"
#hirearchical_clustering.py

bilstm_prepocessed_dataset1_chai = models_dir+"bilstm_preprocessed_dataset1_chai.h5"
lstm_prepocessed_dataset1_chai = models_dir+"lstm_preprocessed_dataset1_chai.h5"
lstm_prepocessed_dataset1_chai_retrain = models_dir+"lstm_preprocessed_dataset1_chai_retrain.h5"

rnn_prepocessed_dataset1_chai = models_dir+"rnn_preprocessed_dataset1_chai.h5"
birnn_prepocessed_dataset1_chai = models_dir+"birnn_preprocessed_dataset1_chai.h5"

gru_prepocessed_dataset1_chai = models_dir+"gru_preprocessed_dataset1_chai.h5"
bigru_prepocessed_dataset1_chai = models_dir+"bigru_preprocessed_dataset1_chai.h5"


lstm_architecture = plots_dir+"lstm_architecture.png"
lstm_architecture_retrain = plots_dir+"lstm_architecture_retrain.png"
bilstm_architecture = plots_dir+"bilstm_architecture.png"

rnn_architecture = plots_dir+"rnn_architecture.png"
birnn_architecture = plots_dir+"birnn_architecture.png"

gru_architecture = plots_dir+"gru_architecture.png"
bigru_architecture = plots_dir+"bigru_architecture.png"

X_val_DL = datasets_dir+"X_val_DL.pkl"
y_val_DL = datasets_dir+"y_val_DL.pkl"

all_classes_performance_lstm = results_dir+"all_classes_performance_lstm.csv"
performance_for_all_DL_models = results_dir+"performance_for_all_DL_models.csv"
deep_learning_dataset_for_retraining = datasets_dir+"deep_learning_dataset_for_retraining.csv"
important_features = datasets_dir+"important_features.csv"



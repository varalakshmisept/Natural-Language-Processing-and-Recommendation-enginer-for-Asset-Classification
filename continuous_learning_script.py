from LSTM_after_correction_train import  lstm_train_after_correction
import config


'''
This script spports retraining through the UI.
'''

def retrain_when_user_prompts():
    return_str,report_dict = lstm_train_after_correction(config.deep_learning_dataset_for_retraining)
    print(return_str,report_dict['accuracy'],report_dict['weighted avg']['precision'],report_dict['weighted avg']['recall'],report_dict['weighted avg']['f1-score'])
    return(round(report_dict['accuracy']*100,2),round(report_dict['weighted avg']['precision']*100,2),round(report_dict['weighted avg']['recall']*100, 2),round(report_dict['weighted avg']['f1-score']*100,2))
    #return(0.40, 0.35, 0.40, 0.34)


if __name__=="__main__":
    retrain_when_user_prompts()


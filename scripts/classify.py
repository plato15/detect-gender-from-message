
import sys, os
import pickle
import pandas as pd
from models import TextSelector, cleaner, DenseTransformer, tokenizer



arguments= sys.argv
print('Here are the arguments ', arguments)
model_file=arguments[1]
text_msg=arguments[2:]
text_msg=" ".join(text_msg)


curr_dir=os.getcwd()
print('Predicting Gender for, ', text_msg)


#run_prediction_and_deploy_model(pred_pipe)
sting_list=[text_msg]
msg_df=pd.DataFrame({'text':sting_list})
print(msg_df)
# load the model from disk and use to predict
print('Looking for ', './model/'+ model_file)
loaded_model = pickle.load(open('./models/'+ model_file, 'rb'))
def predict_text_gender(txt):
    result = loaded_model.predict(txt)
    if result == [1]:
        return 'This message is from a Male'
    else:
        return 'This message is from a Female'


print(predict_text_gender(msg_df))


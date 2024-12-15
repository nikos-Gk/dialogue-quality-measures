import json
import pandas as pd
import os
import sys
import argparse
import time
from stopwords import stopwords
#################################################################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_directory", default="input_directory_with_dialogs",
    help="The folder with the llm dialogs to process")

parser.add_argument(
    "--include_mod_utterances", default=False, type=lambda x: (str(x).lower() == 'true'),
    help="Whether to include the utterances of the moderator")


args = parser.parse_args()

input_directory=args.input_directory
moderator_flag=args.include_mod_utterances

if not os.path.exists(input_directory):
    print(input_directory)
    print("input directory does not exist. Exiting")
    sys.exit(1)

dialogs = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
print("Building corpus of ", len(dialogs), "dialogs")
timestr = time.strftime("%Y%m%d-%H%M%S")
#################################################################################
stopwords=[w.strip() for w in stopwords]
#################################################################################
def processDialog(dialog):
    with open(dialog, 'r', encoding="utf-8") as file:
        data = json.load(file)
    
    conversation_id=data["id"]
    utterances = []
    if not moderator_flag :
        data['logs']=[(log[0],log[1]) for log in data['logs'] if log[0]!="moderator"]
    
    data['logs']=[(log[0],log[1]) for log in data['logs'] if len(log[1])>25]
    for i, log in enumerate(data['logs']):
        speaker, text = log
        text=text.replace('\r\n',' ').replace('\n',' ').strip()
#        text=text.replace("python","")
#        text=text.replace('.',"").strip()
#        text=text.replace('\r\n',' ').replace('\n',' ').strip()
        utterances.append((text,speaker,f"conv_{conversation_id}_utt_{i}"))
    return utterances,conversation_id



def get_metrics_allwords(u1,u2):
    words_u1=[w for w in u1.split()]
    words_u2=[w for w in u2.split()]
    common_words=[w for w in words_u1 if w in words_u2]
    number_of_common_words=len(common_words)
    
    number_of_words_u1=len(words_u1)
    number_of_words_u2=len(words_u2)
    unique_words_of_u1=list(set(words_u1))
    unique_words_of_u2=list(set(words_u2))
    unique_common_words=list(set(unique_words_of_u1+unique_words_of_u2))   


    reply_fraction=float(number_of_common_words)/float(number_of_words_u2)
    op_fraction=float(number_of_common_words)/float(number_of_words_u1)
    jaccard=float((number_of_common_words))/float((len(unique_common_words)))
    
    return (number_of_common_words,reply_fraction,op_fraction,jaccard)
    
   
def get_metrics_stopwords(u1,u2):
    words_u1=[w for w in u1.split() if w in stopwords]
    words_u2=[w for w in u2.split() if w in stopwords]
    common_words=[w for w in words_u1 if w in words_u2]
    number_of_common_words=len(common_words)
    
    number_of_words_u1=len(words_u1)
    number_of_words_u2=len(words_u2)
    unique_words_of_u1=list(set(words_u1))
    unique_words_of_u2=list(set(words_u2))
    unique_common_words=list(set(unique_words_of_u1+unique_words_of_u2))   

    if number_of_words_u2==0:
        reply_fraction=0 
    else:        
        reply_fraction=float(number_of_common_words)/float(number_of_words_u2)
    if number_of_words_u1==0:
        op_fraction=0 
    else:
        op_fraction=float(number_of_common_words)/float(number_of_words_u1)    
    if len(unique_common_words)==0:
        jaccard=0 
    else:
        jaccard=float((number_of_common_words))/float((len(unique_common_words)))
    
    return (number_of_common_words,reply_fraction,op_fraction,jaccard)



def get_metrics_contentwords(u1,u2):
    words_u1=[w for w in u1.split() if w not in stopwords]
    words_u2=[w for w in u2.split() if w not in stopwords]
    common_words=[w for w in words_u1 if w in words_u2]
    number_of_common_words=len(common_words)
    
    number_of_words_u1=len(words_u1)
    number_of_words_u2=len(words_u2)
    unique_words_of_u1=list(set(words_u1))
    unique_words_of_u2=list(set(words_u2))
    unique_common_words=list(set(unique_words_of_u1+unique_words_of_u2))   

    if number_of_words_u2==0:
        reply_fraction=0 
    else:        
        reply_fraction=float(number_of_common_words)/float(number_of_words_u2)
    if number_of_words_u1==0:
        op_fraction=0 
    else:
        op_fraction=float(number_of_common_words)/float(number_of_words_u1)    
    if len(unique_common_words)==0:
        jaccard=0 
    else:
        jaccard=float((number_of_common_words))/float((len(unique_common_words)))
    
    return (number_of_common_words,reply_fraction,op_fraction,jaccard)



language_feat_dict={}
for dialog in dialogs:
    utterances,conversation_id=processDialog(dialog)
    print("language_features-Proccessing dialog: ",conversation_id)
    utt_pair_dict={}
    features_allwords_dict={}    
    features_stopwords_dict={}    
    features_content_dict={}
    
    for i,utt in enumerate(utterances):
        if i==0:
            continue
        utt2_tuple=utterances[i]
        utt1_tuple=utterances[i-1]
        utt2_text_iter, utt2_user_iter,utt2_id_iter=utt2_tuple
        utt1_text_iter, utt1_user_iter,utt1_id_iter=utt1_tuple  
        pair_key=f'pair_{i-1,i}'
        number_of_common_words,reply_fraction,op_fraction,jaccard=get_metrics_allwords(utt1_text_iter,utt2_text_iter)
        number_of_common_stopwords,reply_fraction_stopwords,op_fraction_stopwords,jaccard_stopwords=get_metrics_stopwords(utt1_text_iter,utt2_text_iter)
        number_of_content_words,reply_fraction_content,op_fraction_content,jaccard_content=get_metrics_contentwords(utt1_text_iter,utt2_text_iter)        
        features_allwords_dict={'n_comwords':number_of_common_words,'reply_fra_comwords':reply_fraction,'op_fra_comwords':op_fraction,'jac_comwords':jaccard}
        features_stopwords_dict={'n_stopwords':number_of_common_stopwords,'reply_fra_stopwords':reply_fraction_stopwords,'op_fra_stopwords':op_fraction_stopwords,'jac_stopwords':jaccard_stopwords}
        features_content_dict={'n_contwords':number_of_content_words,'reply_fra_contwords':reply_fraction_content,'op_fra_contwords':op_fraction_content,'jac_contwords':jaccard_content}
        utt_pair_dict[pair_key]=[features_allwords_dict,features_stopwords_dict,features_content_dict]
        language_feat_dict[conversation_id]=utt_pair_dict
        
with open('language_feat_per_pair_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(language_feat_dict, fout, ensure_ascii=False, indent=4)


"""            
with open("language_feat_per_pair_.json", encoding="utf-8") as f:
    lang_scores = json.load(f)
language_feat_dict=lang_scores
"""
        

language_features_per_dialogue={}

number_of_common_words_list=[]
number_of_common_words_list=[]
for conversation_id,dict_iter in language_feat_dict.items():
    total_list=[]
    for pair_key, features_list in dict_iter.items():
        total={}
        for feature in features_list:
            total.update(feature)
        total_list.append(total)
    a=pd.DataFrame(total_list)
    mean=a.mean()
    language_features_per_dialogue[conversation_id]=mean.to_dict()
    

flattened_data = []
for key, value in language_features_per_dialogue.items():
    flattened_data.append(value)
df = pd.DataFrame(flattened_data)
mean_values = df.mean()

language_features_per_dialogue["aggregate_mean"]=mean_values.to_dict()


with open('language_feat_per_dialogue_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(language_features_per_dialogue, fout, ensure_ascii=False, indent=4)

import os
import json
import numpy as np
import sys
import argparse
import time
from transformers import pipeline
pipe=pipeline('text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')
#####################################################################################################################
#####################################################################################################################
#sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
#sys.stderr=io.TextIOWrapper(sys.stderr.buffer,encoding="utf-8")
#####################################################################################################################
#####################################################################################################################
MODERATOR="moderator"
MESSAGE_THREASHOLD_IN_CHARS=25
HARDCODED_MODEL="hardcoded"

parser = argparse.ArgumentParser()
parser.add_argument(
                    "--input_directory", default="input_directory_with_discussions",
                    help="The folder with the llm discussions to process"
                    )

parser.add_argument(
                    "--include_mod_utterances", default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Whether to include the utterances of the moderator"
                    )

args = parser.parse_args()
input_directory=args.input_directory
moderator_flag=args.include_mod_utterances


if not os.path.exists(input_directory):
    print(input_directory)
    print("input directory does not exist. Exiting")
    sys.exit(1)


discussions = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
print("Building corpus of ", len(discussions), "discussions")
timestr = time.strftime("%Y%m%d-%H%M%S")



def processDiscussion(discussion):
    with open(discussion, 'r', encoding="utf-8") as file:
        data = json.load(file)
    
    conversation_id=data["id"]
          
    if not moderator_flag:
        data['logs']=[(log[0],log[1], log[2], log[3]) for log in data['logs'] if log[0]!=MODERATOR]

        
    utterances = []
    
    data['logs']=[
                  (log[0], log[1], log[2], log[3]) 
                  for log in data['logs'] 
                  if ((isinstance(log[1], str)) and (len(log[1]) > MESSAGE_THREASHOLD_IN_CHARS))
                 ]

    for i, log in enumerate(data['logs']):
        speaker, text , model, message_id = log
        text=text.replace('\r\n',' ').replace('\n',' ').rstrip().lstrip()
        if i==0 and model == HARDCODED_MODEL:
            conv_topic=text
        utterances.append((text,speaker,f"conv_{conversation_id}_utt_{i}"))
    return utterances,conversation_id





unorm_disc_cotroversy_dict={}
norm_disc_controversy_dict={}
disc_resultlist_dict={}
disc_productlist_dict={}

for disc in discussions:
    utterances,conversation_id=processDiscussion(disc)
    print("Controversy-Proccessing disc: ",conversation_id)
    utt_resultlist_iter_dict={}
    utt_labelprobproduct_iter_dict={}
    utt_unnormalized_factor_iter_dict={}
    utt_normalized_factor_iter_dict={}
    for utt in utterances:
        try:   
            text_iter, user_iter,id_iter =utt
            if len(text_iter)>512:
                text_iter=text_iter[0:513]
            result_list=pipe([text_iter],return_all_scores=True)[0]
            utt_resultlist_iter_dict[id_iter]=result_list
            product_list=[]
            for result in result_list:
                star=int(result['label'][0])
                prob=float(result['score'])
                product=star*prob
                product_list.append(product)
            utt_labelprobproduct_iter_dict[id_iter]=product_list        
            sent_score=sum(product_list)
            utt_unnormalized_factor_iter_dict[id_iter]=sent_score
        except Exception as ex:
            print("Exception at atterance: ",utt)
            print(ex)
    disc_resultlist_dict[conversation_id]=utt_resultlist_iter_dict
    disc_productlist_dict[conversation_id]=utt_labelprobproduct_iter_dict
    normalized_factor=sum(utt_unnormalized_factor_iter_dict.values())
    utt_normalized_factor_iter_dict={key: value/normalized_factor for key,value in utt_unnormalized_factor_iter_dict.items()}
    unorm_disc_cotroversy_dict[conversation_id]=utt_unnormalized_factor_iter_dict
    norm_disc_controversy_dict[conversation_id]=utt_normalized_factor_iter_dict

with open('bert_sent_per_comment_unorm_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(unorm_disc_cotroversy_dict, fout, ensure_ascii=False, indent=4)

with open('bert_sent_per_comment_norm_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(norm_disc_controversy_dict, fout, ensure_ascii=False, indent=4)

"""            
    with open("bert_sent_per_comment_unorm_.json", encoding="utf-8") as f:
        sent_scores = json.load(f)
    unorm_disc_cotroversy_dict=sent_scores
"""

"""            
    with open("bert_sent_per_comment_norm_.json", encoding="utf-8") as f:
        sent_scores = json.load(f)
    norm_disc_controversy_dict=sent_scores
"""

controversy_per_disc_unorm={}

for conversation_id,dict_iter in unorm_disc_cotroversy_dict.items():
    std_iter=np.std(list(dict_iter.values()),ddof=1)
    controversy_per_disc_unorm[conversation_id]=std_iter

controversy_scores_mean=np.mean([float(number) for number in list(controversy_per_disc_unorm.values())])
controversy_per_disc_unorm["aggregate_mean"]=str(controversy_scores_mean)


with open('controversy_per_disc_unorm_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(controversy_per_disc_unorm, fout, ensure_ascii=False, indent=4)

controversy_per_disc_norm={}

for conversation_id,dict_iter in norm_disc_controversy_dict.items():
    std_iter=np.std(list(dict_iter.values()),ddof=1)
    controversy_per_disc_norm[conversation_id]=std_iter

controversy_scores_mean=np.mean([float(number) for number in list(controversy_per_disc_norm.values())])
controversy_per_disc_norm["aggregate_mean"]=str(controversy_scores_mean)


with open('controversy_per_disc_norm_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(controversy_per_disc_norm, fout, ensure_ascii=False, indent=4)

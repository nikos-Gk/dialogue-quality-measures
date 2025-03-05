import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import io
import sys
import argparse
import json
from convokit import Corpus, Utterance, Speaker, HyperConvo
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
#################################################################################
#sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
#sys.stderr=io.TextIOWrapper(sys.stderr.buffer,encoding="utf-8")
#################################################################################
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


def assign_sentence_to_speakers(sentence_text,speakers,previous_speaker):
    assigned_speakers=[]
    for speaker in speakers.keys():
        if speaker in sentence_text:
            assigned_speakers.append(speaker)
    if len(assigned_speakers) == 0:
        assigned_speakers=previous_speaker
    return assigned_speakers


def processDiscussion(discussion):
    with open(discussion, 'r', encoding="utf-8") as file:
        data = json.load(file)
        
    conversation_id=data["id"]
    print(f"conversation_id:{conversation_id}")
    speakers = {user: Speaker(id=user) for user in data['users']}

    if not moderator_flag:
        data['logs']=[(log[0],log[1], log[2], log[3]) for log in data['logs'] if log[0]!=MODERATOR]
        
    data['logs']=[
                  (log[0], log[1], log[2], log[3]) 
                  for log in data['logs'] 
                  if ((isinstance(log[1], str)) and (len(log[1]) > MESSAGE_THREASHOLD_IN_CHARS))
                 ]
 
    utterances = []
    utterances_history={}
    reply_to_dict={}
    data_cord={"logs":[]}
    data_cord_counter=-1
    for i, log in enumerate(data['logs']):
        current_speaker,original_text,model,mid=log
        if i == 0:
            data_cord["logs"].append((current_speaker,original_text,model,mid,"-"))
            data_cord_counter+=1
            continue
        sentences = sent_tokenize(original_text)
        sentence_dict={}
        for sentence_index,sentence_text in enumerate(sentences):
            if sentence_index==0:
                assigned_speakers=assign_sentence_to_speakers(sentence_text,speakers,[])
            else:
                assigned_speakers=assign_sentence_to_speakers(sentence_text,speakers,assigned_speakers)
            sentence_dict[sentence_index]=assigned_speakers
            
        users_text={}
        for sentence_index,sentence_text in enumerate(sentences):
            assinged_users=sentence_dict[sentence_index]
            
            if len(assinged_users)==0:
                previous_speaker, text, model, mid, replyto_user=data_cord['logs'][data_cord_counter]
                assinged_users.append(previous_speaker)  
                    
            for user in assinged_users:
                if user in users_text.keys():
                    users_text[user]=users_text[user]+"\n"+sentence_text
                else:
                    users_text[user]=sentence_text
        
        for user,text in users_text.items():
            data_cord["logs"].append((current_speaker,text,model,mid,user)) 
            data_cord_counter+=1
    
    
    
    data_cord['logs']=[
                          (log[0], log[1], log[2], log[3] , log[4])
                          for log in data_cord['logs'] 
                          if log[0] != log[4]  #disregard users who reply to themselves
                      ]
    
    strTime = time.strftime("%Y%m%d-%H%M%S")
    tm=datetime.strptime(strTime,"%Y%m%d-%H%M%S")
    for i, log in enumerate(data_cord['logs']):
        speaker, text, model, mid, replyto_user = log
        utterances_history[speaker]=i
        tm=tm+relativedelta(seconds=1)
        if i==0:          
          reply_to_id=None
        else:
            utt_id=utterances_history[replyto_user]
            reply_to_id="utt_"+str(utt_id)+'_'+str(conversation_id)
            
        g=Utterance(id=f"utt_{i}_{conversation_id}",
                   speaker=speakers[speaker],
                   conversation_id=str(conversation_id),
                   reply_to=reply_to_id,
                   text=text,
                   #meta={'timestamp': data['timestamp']})
                   meta={'timestamp': tm}
                   )
        g.timestamp=tm
        utterances.append(g)
        reply_to_dict[i]=reply_to_id
    return utterances,speakers,data,utterances_history,data_cord,conversation_id,reply_to_dict

aggregate_utterances=[]
aggregate_datacord={}
output_dict={}
for disc in discussions:
    utterances,speakers,data,utterances_history,data_cord,conversation_id,reply_to_dict=processDiscussion(disc)
    aggregate_datacord[conversation_id]=data_cord
    aggregate_utterances=aggregate_utterances+utterances



corpus = Corpus(utterances=aggregate_utterances)
print("Corpus created successfully.")
corpus.print_summary_stats()  

#############################################################################################################
#prefix_len – Use the first [prefix_len] utterances of each conversation to construct the hypergraph
#min_convo_len – Only consider conversations of at least this length
hc = HyperConvo(prefix_len=40, min_convo_len=2) 
hc.transform(corpus)
#############################################################################################################
dt_features=corpus.get_vector_matrix('hyperconvo').to_dataframe()
dt_features = dt_features.fillna(-1)
feat_names = list(dt_features.columns)
motif_count_feats = [x for x in feat_names if ('count' in x) and ('mid' not in x)]
#############################################################################################################
motifs_dt=dt_features[motif_count_feats]
motifs_dt_tp=motifs_dt.transpose()
motifs_dict=motifs_dt_tp.to_dict()
mean_per_motif=motifs_dt.mean()
m=mean_per_motif.to_dict()
motifs_dict['aggregate_mean']=m
#############################################################################################################
dt_features_tp=dt_features.transpose()
features_dict=dt_features_tp.to_dict()
mean_per_feature=dt_features.mean()
f=mean_per_feature.to_dict()
features_dict['aggregate_mean']=f
#############################################################################################################
with open('reciprocity_motifs_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(motifs_dict, fout, ensure_ascii=False, indent=4)
#############################################################################################################
with open('structure_features_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(features_dict, fout, ensure_ascii=False, indent=4)
#############################################################################################################
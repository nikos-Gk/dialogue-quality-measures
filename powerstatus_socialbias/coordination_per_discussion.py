import os
import sys
import argparse
import json
from convokit import Corpus, Utterance, Speaker, Coordination
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
#import nltk
from nltk.tokenize import sent_tokenize
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
        
    has_moderator=False
    conversation_id=data["id"]
    print(f"conversation_id:{conversation_id}")
    speakers = {user: Speaker(id=user) for user in data['users']}
    
    if MODERATOR in speakers.keys():
        has_moderator=True

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
    return utterances,speakers,data,utterances_history,data_cord,conversation_id,reply_to_dict,has_moderator

aggregate_utterances=[]
aggregate_datacord={}
output_dict={}
for disc in discussions:
    utterances,speakers,data,utterances_history,data_cord,conversation_id,reply_to_dict,has_moderator=processDiscussion(disc)
    aggregate_datacord[conversation_id]=data_cord
    aggregate_utterances=aggregate_utterances+utterances
    corpus = Corpus(utterances=utterances)
    print("Corpus created successfully.")
    corpus.print_summary_stats()
    coord = Coordination()
    coord.fit(corpus)
    coord.transform(corpus)
    all_speakers=lambda speaker:True
    user_lamda_list=[]

    if has_moderator and moderator_flag:
        moderators=lambda speaker: speaker.id==MODERATOR
        user_lamda_list.append({'moderator': (moderators)})
    for person in speakers.keys():
        if person==MODERATOR:
            continue
        user_lamda_list.append({person: (lambda speaker, person=person: speaker.id == person)})
    print("coordination of all speakers to user")
    coord_allspeakers_2_user={}
    for item in user_lamda_list:
        user=list(item.keys())[0]
        lamda_function=list(item.values())[0]
        print("> coordination of all speakers to user ",user)
        allspeakers_2_user=coord.summarize(corpus,all_speakers,lamda_function,focus="targets",target_thresh=1,summary_report=True)
        coord_allspeakers_2_user[user]=allspeakers_2_user
    print("coordination of user to all speakers")
    coord_user_2_allspeaker={}
    for item in user_lamda_list:
        user=list(item.keys())[0]
        lamda_function=list(item.values())[0]
        print(f"> coordination of user {user} to all speakers ")
        user_2_allspeakers=coord.summarize(corpus,lamda_function,all_speakers,focus="speakers",speaker_thresh=1,target_thresh=1,summary_report=True)
        coord_user_2_allspeaker[user]=user_2_allspeakers
    output_dict[conversation_id]={'coord_allspeakers_2_user':coord_allspeakers_2_user,'coord_user_2_allspeaker':coord_user_2_allspeaker}

    

with open('coord_per_discussion_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(output_dict, fout, ensure_ascii=False, indent=4)

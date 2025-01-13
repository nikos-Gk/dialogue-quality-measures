import os
import sys
import argparse
import json
from convokit import Corpus, Utterance, Speaker, Coordination
import time

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

def find_utterance_replied_to(current_speaker,current_utterance_id,text,speakers,utterances_history):
    for person,speaker_value in speakers.items():
        if person != current_speaker and person in text:
            if person in utterances_history.keys():
                person_last_utterance_id=utterances_history[person]
                return person_last_utterance_id
            else:
                current_utterance_id-1
    return current_utterance_id-1
    

def processDialog(dialog):
    with open(dialog, 'r', encoding="utf-8") as file:
        data = json.load(file)
        
    conversation_id=data["id"]
    print(f'conversation id, {conversation_id}')
    has_moderator=True
    if not data["moderator"]:
        has_moderator=False
    
    speakers = {
        user: Speaker(id=user, meta={'type': user_type}) for user, user_type in zip(data['users'], data['user_types'])
    }
    
        
    if has_moderator and moderator_flag:
        speakers[data['moderator']] = Speaker(id=data['moderator'], meta={'type': data['moderator_type']})

    
    parts=data['user_prompts'][0].split("You see the following post on a social media site:")
    conv_topic=parts[1].split('Your instructions:')[0].strip()
   
    utterances_history={}
    utterances = []
    
    if not moderator_flag:
        data['logs']=[(log[0],log[1]) for log in data['logs'] if log[0]!="moderator"]

    data['logs']=[(log[0],log[1]) for log in data['logs'] if len(log[1])>25]   

    data_cord={"logs":[]}
    for i, log in enumerate(data['logs']):
        speak_loc_dict={}
        speaker, text = log
        counter=0
        for person,speaker_value in speakers.items():
            if person!= speaker and person in text and (". @"+person in text or text.index(person)<50 ):
                start=text.index(person)
                counter+=1
                speak_loc_dict[person]=[start]
        if counter>1:
            sorted_speak_loc_dict=dict(sorted(speak_loc_dict.items(), key=lambda item: item[1]))
            sorted_values_l=list(sorted_speak_loc_dict.values())
            for i,v in enumerate(sorted_values_l):
                if i==0:
                    start_index=0
                    end_index=sorted_values_l[i+1][0]
                    text_iter=text[start_index:end_index]
                    data_cord['logs'].append((speaker,text_iter))
                elif i in range(1,len(sorted_values_l)-1):
                    start_index=end_index
                    end_index=sorted_values_l[i+1][0]
                    text_iter=text[start_index:end_index]
                    data_cord['logs'].append((speaker,text_iter))
                else:
                    start_index=end_index
                    text_iter=text[start_index:]
                    data_cord['logs'].append((speaker,text_iter))
        else:
           data_cord['logs'].append((speaker,text))
    
    for i, log in enumerate(data_cord['logs']):
        speaker, text = log
        utt_id=find_utterance_replied_to(speaker,i,text,speakers,utterances_history)
        if i==0:
            reply_to_id=None
        else:
            reply_to_id="utt_"+str(utt_id)+'_'+str(conversation_id)
        utterances.append(
            Utterance(id=f"utt_{i}_{conversation_id}",
                      speaker=speakers[speaker],
                      conversation_id=str(conversation_id),
                      reply_to=reply_to_id,
                      text=text,
                      meta={'timestamp': data['timestamp']})
            )
        utterances_history[speaker]=i
    return utterances,speakers,data,utterances_history,data_cord,conversation_id,has_moderator

aggregate_utterances=[]
aggregate_datacord={}
output_dict={}
for dialog in dialogs:
    utterances,speakers,data,utterances_history,data_cord,conversation_id,has_moderator=processDialog(dialog)
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
        moderators=lambda speaker: speaker.id=='moderator'
        user_lamda_list.append({'moderator': (moderators)})
    for person in speakers.keys():
        if person=='moderator':
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
    
###################################################################################
#with open('coord_per_dialogue_'+timestr+'.json', 'w',encoding="utf-8") as fout:
#    json.dump(output_dict, fout, ensure_ascii=False, indent=4)
###################################################################################
###################################################################################
#with open('aggregate_datacord_'+timestr+'.json', 'w',encoding="utf-8") as fout:
#    json.dump(aggregate_datacord, fout, ensure_ascii=False, indent=4)
###################################################################################

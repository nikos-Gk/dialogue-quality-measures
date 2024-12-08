import os
import sys
import argparse
import json
from convokit import Corpus, Utterance, Speaker
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import math

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
            if person not in utterances_history:
                return 0
            person_last_utterance_id=utterances_history[person]
            return person_last_utterance_id
    return current_utterance_id-1
    

def processDialog(dialog):
    with open(dialog, 'r', encoding="utf-8") as file:
        data = json.load(file)
    
    conversation_id=data["id"]
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


    
    for i, log in enumerate(data['logs']):
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
    return utterances, conv_topic, speakers


dialogue_speakers_messages_dict={}
dialogue_number_of_messages_dict={}
dialogue_sum_number_of_words_dict={}
for dialog in dialogs:
    utterances, conv_topic, speakers=processDialog(dialog)
    dialog_id=utterances[0].id.split("_")[2]
    print("Balanced Participation-Proccessing dialog: ",dialog_id," with LLM")
    dialogue_speakers_messages_dict[dialog_id]={speaker:[] for speaker in speakers.keys()}
    dialogue_number_of_messages_dict[dialog_id]=[]
    dialogue_sum_number_of_words_dict[dialog_id]=[]
    for utt in utterances:
        speaker=utt.get_speaker().id
        dialogue_speakers_messages_dict[dialog_id][speaker].append(utt.text)
    for speaker in dialogue_speakers_messages_dict[dialog_id].keys():
        speaker_words_per_message=[]
        speaker_number_of_messages=len(dialogue_speakers_messages_dict[dialog_id][speaker])
        dialogue_number_of_messages_dict[dialog_id].append(speaker_number_of_messages)
        for message in dialogue_speakers_messages_dict[dialog_id][speaker]:
            speaker_words_per_message.append(len(message.split()))
        dialogue_speaker_number_of_words=sum(speaker_words_per_message)
        dialogue_sum_number_of_words_dict[dialog_id].append(dialogue_speaker_number_of_words)  


with open('number_of_messages_per_dialog_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(dialogue_number_of_messages_dict, fout, ensure_ascii=False, indent=4)

with open('sum_of_words_per_dialog_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(dialogue_sum_number_of_words_dict, fout, ensure_ascii=False, indent=4)

def compute_balance(contributions):
    """
    Compute the balance (S) of conversational contributions using entropy.
    
    Parameters:
    contributions (list): A list of contributions by each participant.
    
    Returns:
    float: The balance (S) of the conversation.
    """
    # Calculate the total contributions
    total_contributions = sum(contributions)
    
    # Calculate the proportion of contributions for each participant
    proportions = [c / total_contributions for c in contributions]
    
    # Compute the entropy (balance)
    balance = -sum(p * math.log(p,len(proportions)) for p in proportions if p > 0)
    
    return balance


#entropy_per_dialogue={dialogue_id:{'entropy_messages':compute_balance(dialogue_number_of_messages_dict[dialogue_id]),'entropy_number_of_words':compute_balance(dialogue_sum_number_of_words_dict[dialogue_id])} for dialogue_id in dialogue_speakers_messages_dict.keys()}

entropy_per_dialogue={}
for dialogue_id in dialogue_speakers_messages_dict.keys():
    entropy_per_dialogue[dialogue_id]={}
    entropy_per_dialogue[dialogue_id]['entropy_number_of_messages']=compute_balance(dialogue_number_of_messages_dict[dialogue_id])
    entropy_per_dialogue[dialogue_id]['entropy_number_of_words']=compute_balance(dialogue_sum_number_of_words_dict[dialogue_id])

"""    
with open('entropy_per_dialogue_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(entropy_per_dialogue, fout, ensure_ascii=False, indent=4)
"""

flattened_data_number_of_messages=[]
flattened_data_number_of_words=[]
for entropy_features_dict in entropy_per_dialogue.values():
    for feature_entropy, value in entropy_features_dict.items():
        if feature_entropy=='entropy_number_of_messages':
            flattened_data_number_of_messages.append(value)
        elif feature_entropy=='entropy_number_of_words':
            flattened_data_number_of_words.append(value)
        else:
            print('error in flattened data')
            break
df = pd.DataFrame({'entropy_number_of_messages':flattened_data_number_of_messages,'entropy_number_of_words':flattened_data_number_of_words})
mean_values = df.mean()

entropy_per_dialogue["aggregate_mean"]=mean_values.to_dict()

with open('entropy_per_dialogue_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(entropy_per_dialogue, fout, ensure_ascii=False, indent=4)



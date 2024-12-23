# -*- coding: utf-8 -*-
import json
import pandas as pd
import os
import sys
import argparse
import time
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
        utterances.append((text,speaker,f"conv_{conversation_id}_utt_{i}"))
    return utterances,conversation_id



all_dialogs=[]
for dialog in dialogs:
    utterances,conversation_id=processDialog(dialog)
    all_dialogs.append(utterances)
    
rows=[]

for dialog in all_dialogs:
    for utt in dialog:
        text,user,utt_id=utt
        conversation_id=utt_id.split("_")[1]
        utt_number='utt_'+utt_id.split("_")[3]
        rows.append({"en_text":text,"conversation_id":conversation_id,"utt_id":utt_number,"user":user})

df=pd.DataFrame(rows)

print("Saving .csv file...")
df.to_csv(timestr+'.csv',sep='\t',index=False)
        
   
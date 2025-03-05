# -*- coding: utf-8 -*-
import json
import pandas as pd
import os
import sys
import argparse
import time
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
#################################################################################
def processDiscussion(discussion):
    with open(discussion, 'r', encoding="utf-8") as file:
        data = json.load(file)
    
    conversation_id=data["id"]
    utterances = []

    if not moderator_flag:
        data['logs']=[(log[0],log[1], log[2], log[3]) for log in data['logs'] if log[0]!=MODERATOR]
        
    data['logs']=[
                  (log[0], log[1], log[2], log[3]) 
                  for log in data['logs'] 
                  if ((isinstance(log[1], str)) and (len(log[1]) > MESSAGE_THREASHOLD_IN_CHARS))
                 ]


    for i, log in enumerate(data['logs']):
        speaker, text , model, message_id = log
        text=text.replace('\r\n',' ').replace('\n',' ').rstrip().lstrip()
        utterances.append((text,speaker,f"conv_{conversation_id}_utt_{i}"))
    return utterances,conversation_id





aggregate_utterances=[]
for disc in discussions:
    utterances,conversation_id=processDiscussion(disc)
    aggregate_utterances.append(utterances)


rows=[]

for disc in aggregate_utterances:
    for utt in disc:
        text,user,utt_id=utt
        conversation_id=utt_id.split("_")[1]
        utt_number='utt_'+utt_id.split("_")[3]
        rows.append({"en_text":text,"conversation_id":conversation_id,"utt_id":utt_number,"user":user})

df=pd.DataFrame(rows)

print("Saving .csv file...")
df.to_csv(timestr+'.csv',sep='\t',index=False)
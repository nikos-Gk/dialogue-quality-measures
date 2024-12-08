# -*- coding: utf-8 -*-
import openai
import time
import os
import sys
import argparse
import json
from convokit import Utterance, Speaker
import numpy as np

prompt="""{conv_text}'\n\n\n

The texts above show a dialogue in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating 
inflammatory and aggressive speech.
Now, please use chain-of-thought reasoning to rate the informativeness of the above discussion.
After the Chain-of-Thoughts reasoning steps, you should assign a score for the informativeness of the entire discussion on a scale of 1 to 5, where 1 is of poor information quality (uninformative) and 5 of high information quality (informative). 
Conclude your evaluation with the statement: 'The informativeness of the comments presented in the above discussion is: [X]', where X is the numeric score (real number) you've determined. 
Please, ensure that your last statement is the score in brackets [].
"""




parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_directory", default="input_directory_with_dialogs",
    help="The folder with the llm dialogs to process")

parser.add_argument(
    "--openAIKEY", default="",
    help="Open AI API key")


args = parser.parse_args()

input_directory=args.input_directory
openAIKEY=args.openAIKEY




if not os.path.exists(input_directory):
    print(input_directory)
    print("input directory does not exist. Exiting")
    sys.exit(1)
    
if not openAIKEY:
    print("OpenAI API key does not exist. Informativeness scores will not be computed. Exiting")   
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
    
    if has_moderator:
        speakers[data['moderator']] = Speaker(id=data['moderator'], meta={'type': data['moderator_type']})

    parts=data['user_prompts'][0].split("You see the following post on a social media site:")
    conv_topic=parts[1].split('Your instructions:')[0].strip()
        
    utterances_history={}
    utterances = []
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
    return utterances,conv_topic


def prompt_gpt4(prompt,key):
    openai.api_key = key    
    ok = False
    counter=0
    while not ok:  # to avoid "ServiceUnavailableError: The server is overloaded or not ready yet."
        counter=counter+1
        try:
            response = openai.ChatCompletion.create(
                        model="gpt-4-1106-preview",
                        messages=[
                                  {"role": "user", "content": prompt}
                                  ],
                        max_tokens = 4096,
                        temperature = 0
                        )
            ok = True                    
        except Exception as ex:
            print("error", ex)
            print("sleep for 5 seconds")
            time.sleep(5)
            if counter>10:
                ok=False
                return -1
    return response["choices"][0]["message"]["content"]


def calculate_dialogue_informativeness_score(utts,topic,key):
    conv_text=""
    for utt in utts:
        text=utt.text
        speaker=utt.get_speaker().id
        conv_text=conv_text+speaker+": "+text+"\n\n"
        formatted_prompt=prompt.format(conv_text=conv_text,post=topic)
    annotations_ci=[]
    try:
        response_text = prompt_gpt4(formatted_prompt,key)
#        print(formatted_prompt)
        annotations_ci.append(response_text)
    except Exception as e:  
        print('Error: ', e)
        annotations_ci.append(-1)
    return annotations_ci


if openAIKEY :
    inform_scores_llm_output_dict={}
    for dialog in dialogs:
        utterances,conv_topic=processDialog(dialog)
        dialog_id=utterances[0].id.split("_")[2]
        print("Overall Informativeness Score-Proccessing dialog: ",dialog_id," with LLM")
        inform_score=calculate_dialogue_informativeness_score(utterances,conv_topic,openAIKEY)
        inform_scores_llm_output_dict[dialog_id]=inform_score


    with open('llm_output_informativeness_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(inform_scores_llm_output_dict, fout, ensure_ascii=False, indent=4)




    """        
    with open("llm_output_informativeness_.json", encoding="utf-8") as f:
        info_scores = json.load(f)
    inform_scores_llm_output_dict=info_scores
    """

    inform_scores_per_dialog={}
    for dialog_id, turnAnnotations in inform_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label==-1:
                print("LLM output with missing overall informativeness score , skipping dialogue\n")
                print(label)
                continue
            parts=label.split("informativeness of the comments presented in the above discussion is:")
            value=parts[1]
            rightParenthesisIndex=value.find("]")
            value=value[0:rightParenthesisIndex]
            value=value.replace("[","").replace("]","")
            inform_scores_per_dialog[dialog_id]=value

    inform_scores_mean=np.mean([float(number) for number in list(inform_scores_per_dialog.values())])
    
    inform_scores_per_dialog["aggregate_mean"]=str(inform_scores_mean)
    
    
    with open('inform_scores_per_dialogue_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(inform_scores_per_dialog, fout, ensure_ascii=False, indent=4)

 
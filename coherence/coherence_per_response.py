# -*- coding: utf-8 -*-
import openai
import time
import os
import sys
import argparse
import json
from convokit import Utterance, Speaker
import pandas as pd
import numpy as np

prompt= """\n\n
You will be presented with a conversation history (which can be empty if the new utterance is the first utterance made in the conversation) from a 
dialogue in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating 
inflammatory and aggressive speech.

*CONVERSATION HISTORY*: "{conv_history}"

*RESPONSE*: "{response}"

You are a coherence evaluator. 
Given the post that the discussion is based on and the conversation history, you have to assign a score on a scale of 1 to 5 that indicates the coherence of the reponse.
A score 1 indicates that the response is of poor coherence quality (incoherent), while 5 indicates that the response is of extremely high coherence quality (coherent).
Noteworthy, the conversation history is provided for you to simply understand the utterances made before the new utterance so as to help you better 
annotate the coherence of the new response.
Please provide the final answer directly with no reasoning steps.
For clarity, your evaluation should be presented with the statement: 'The coherence of the new response is: [X]', where X is the numeric score (real number) you've determined. 
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
    print("OpenAI API key does not exist. Coherence scores will not be computed. Exiting")    
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


def calculate_response_coherence_score(utts,topic,key):
    conv_hist = ''
    annotations_ci=[]
    for index,utt in enumerate(utts):
        text=utt.text
        speaker=utt.get_speaker().id  
        if index==0:
            conv_hist=''
        else:
            conv_hist=conv_hist+'\n'+ "<user_name="+utts[index-1].get_speaker().id+"\n"+utts[index-1].text+'\n'
        try:
            formatted_prompt = prompt.format(response="<user_name="+speaker+">" + '\n' + text, conv_history=conv_hist, post=topic)
            response_text = prompt_gpt4(formatted_prompt,openAIKEY)
#            print(formatted_prompt)
            annotations_ci.append(response_text)
        except Exception as e: # happens for one instance 
            print('Error: ', e)
            annotations_ci.append(-1)
    return annotations_ci

if openAIKEY :
    coh_per_resp_scores_llm_output_dict={}
    for dialog in dialogs:
        try:
            utterances,conv_topic=processDialog(dialog)
            dialog_id=utterances[0].id.split("_")[2]
            print("Coherence Score Per Response-Proccessing dialog: ",dialog_id," with LLM")
            coh_per_resp=calculate_response_coherence_score(utterances,conv_topic,openAIKEY)
            coh_per_resp_scores_llm_output_dict[dialog_id]=coh_per_resp
            print("Sleeping for 60 seconds, for openAI quota")
            time.sleep(60)
        except Exception as e: 
             print('Error: ', e)
             print(dialog)

    with open('llm_output_coh_per_response_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(coh_per_resp_scores_llm_output_dict, fout, ensure_ascii=False, indent=4)



    """            
        with open("llm_output_coh_per_response_.json", encoding="utf-8") as f:
            coh_scores = json.load(f)
        coh_per_resp_scores_llm_output_dict=coh_scores
    """

    coherence_scores_per_response={}
    for dialog_id, turnAnnotations in coh_per_resp_scores_llm_output_dict.items():
        counter=0
        ut_dict={}
        for label in turnAnnotations:
            if label==-1:
                print("LLM output with missing coherence response score , skipping response\n")
                print(label)
                counter+=1
                continue
            parts=label.split("coherence of the new response is:")
            value=parts[1]
            rightParenthesisIndex=value.find("]")
            value=value[0:rightParenthesisIndex]
            value=value.replace("[","").replace("]","")
            ut_dict[dialog_id+"_"+str(counter)]=value
            counter+=1
        coherence_scores_per_response[dialog_id]=ut_dict

    
    average_coherence_per_dialog=[]
    for dialog_id, utt_coh in coherence_scores_per_response.items():
        utt_coh_per_dialog_df=pd.DataFrame(utt_coh,index=[0])
        utt_coh_per_dialog_df=utt_coh_per_dialog_df.transpose()
        cols = utt_coh_per_dialog_df.columns
        utt_coh_per_dialog_df[cols] = utt_coh_per_dialog_df[cols].apply(pd.to_numeric, errors='coerce')
        mean=utt_coh_per_dialog_df.mean()
        average_coherence_per_dialog.append({dialog_id:mean.to_dict()[0]})

    # Flatten the list of dictionaries
    flattened_data = []
    for item in average_coherence_per_dialog:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()
    
    average_coherence_per_dialog.append({"aggregate_mean":mean_values.to_dict()[0]})

    with open('dialogue_aver_coh_per_response_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(average_coherence_per_dialog, fout, ensure_ascii=False, indent=4)
        
    with open('coherence_per_response_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(coherence_scores_per_response, fout, ensure_ascii=False, indent=4)

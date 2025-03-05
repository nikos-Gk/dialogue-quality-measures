# -*- coding: utf-8 -*-
import openai
import time
import os
import io
import sys
import argparse
import json
from convokit import Utterance, Speaker
import numpy as np
#################################################################################
#sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
#sys.stderr=io.TextIOWrapper(sys.stderr.buffer,encoding="utf-8")
#################################################################################
prompt="""{conv_text}'\n\n\n

The texts above show a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating inflammatory and aggressive speech.
Now, please use chain-of-thought reasoning to rate the engagement of the above discussion.
Engagement refers to the degree to which a discussion sustains interest and participation.
After the Chain-of-Thoughts reasoning steps, rate the engagement of the entire discussion on a scale of 1 to 5, where 1 is of poor engagement quality (that is the comments do not sustain interest and participation) and 5 of high engagement quality (comments sustain interest and participation to a high degree). 
Conclude your evaluation with the statement: 'The engagement quality of the above discussion is: [X]', where X is the rating you've determined. 
Please, ensure that your last statement is the score in brackets [].
"""
MODERATOR="moderator"
MESSAGE_THREASHOLD_IN_CHARS=25
HARDCODED_MODEL="hardcoded"


parser = argparse.ArgumentParser()
parser.add_argument(
                    "--input_directory", default="input_directory_with_discussions",
                    help="The folder with the llm discussions to process"
                    )

parser.add_argument(
                    "--openAIKEY", default="",
                    help="Open AI API key"
                    )

parser.add_argument(
                    "--include_mod_utterances", default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Whether to include the utterances of the moderator"
                    )

args = parser.parse_args()
input_directory=args.input_directory
openAIKEY=args.openAIKEY
moderator_flag=args.include_mod_utterances


if not os.path.exists(input_directory):
    print(input_directory)
    print("input directory does not exist. Exiting")
    sys.exit(1)
    
if not openAIKEY:
    print("OpenAI API key does not exist. Engagement scores will not be computed. Exiting")    
    sys.exit(1)


discussions = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
print("Building corpus of ", len(discussions), "discussions")
timestr = time.strftime("%Y%m%d-%H%M%S")

def processDiscussion(discussion):
    with open(discussion, 'r', encoding="utf-8") as file:
        data = json.load(file)    

    conversation_id=data["id"]
    speakers = {user: Speaker(id=user) for user in data['users']}

    if not moderator_flag:
        data['logs']=[(log[0],log[1], log[2], log[3]) for log in data['logs'] if log[0]!=MODERATOR]
    utterances = []
    data['logs']=[
                  (log[0], log[1], log[2], log[3]) 
                  for log in data['logs'] 
                  if ((isinstance(log[1], str)) and (len(log[1]) > MESSAGE_THREASHOLD_IN_CHARS))]

    for i, log in enumerate(data['logs']):
        speaker, text , model, message_id = log
        text=text.rstrip().lstrip()
        if i==0 and model == HARDCODED_MODEL:
            conv_topic=text
        utterances.append(
                          Utterance(id=f"utt_{i}_{conversation_id}",
                                  speaker=speakers[speaker],
                                  conversation_id=str(conversation_id),
                                  text=text,
                                  meta={'timestamp': data['timestamp']}))
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
                return -1
    return response["choices"][0]["message"]["content"]


def calculate_discussion_engagement_score(utts,topic,key):
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
    engag_scores_llm_output_dict={}
    for disc in discussions:
        utterances,conv_topic=processDiscussion(disc)
        disc_id=utterances[0].id.split("_")[2]
        print("Overall Engagement Score-Proccessing discussion: ",disc_id," with LLM")
        engag_score=calculate_discussion_engagement_score(utterances,conv_topic,openAIKEY)
        engag_scores_llm_output_dict[disc_id]=engag_score




    
    with open('llm_output_engagement_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(engag_scores_llm_output_dict, fout, ensure_ascii=False, indent=4)
    



    """        
    with open("llm_output_engagement_.json", encoding="utf-8") as f:
        engag_scores = json.load(f)
    engag_scores_llm_output_dict=engag_scores
    """

    engag_scores_per_disc={}
    for disc_id, turnAnnotations in engag_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label==-1:
                print("LLM output with missing overall engagement score , skipping discussion\n")
                print(label)
                continue
            parts=label.split("engagement quality of the above discussion is:")
            value=parts[1]
            rightParenthesisIndex=value.find("]")
            
            leftParenthesisIndex=value.find("[")
            value=value[leftParenthesisIndex:rightParenthesisIndex]
            value=value.replace("[","").replace("]","")
            engag_scores_per_disc[disc_id]=value

    engag_scores_mean=np.mean([float(number) for number in list(engag_scores_per_disc.values())])
    
    engag_scores_per_disc["aggregate_mean"]=str(engag_scores_mean)
    
    
    with open('engag_scores_per_discussion_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(engag_scores_per_disc, fout, ensure_ascii=False, indent=4)

import os
import sys
import argparse
import json
from convokit import Corpus, Utterance, Speaker, PolitenessStrategies, TextParser
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from collaboration import Collaboration
from dispute_tactics import DisputeTactics
from argument_quality_overall import OAQuality
from argument_quality_per_dimensions import AQualityDimensions

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_directory", default="input_directory_with_dialogs",
    help="The folder with the llm dialogs to process")
parser.add_argument(
    "--calculate_politeness", default=False, type=lambda x: (str(x).lower() == 'true'),
    help="Whether to calculate politeness or not")
parser.add_argument(
    "--openAIKEY", default="",
    help="Open AI API key")
parser.add_argument(
    "--calculate_dispute_tactics", default=False, type=lambda x: (str(x).lower() == 'true'),
    help="Wether to calculate dispute tactics")

parser.add_argument(
    "--prompt_mode", default='real',
    help="mode of prompt for the average overall argument quality score. Permited values: real, rating")

parser.add_argument(
    "--calculate_maq", default=False, type=lambda x: (str(x).lower() == 'true'),
    help="Whether to calculate multi dimensional argument quality")

parser.add_argument(
    "--calculate_oaq", default=False, type=lambda x: (str(x).lower() == 'true'),
    help="Whether to calculate overall argument quality")


args = parser.parse_args()

input_directory=args.input_directory
politeness_flag=args.calculate_politeness
calculate_dispute_tactics=args.calculate_dispute_tactics
openAIKEY=args.openAIKEY
mode=args.prompt_mode
calculate_maq=args.calculate_maq
calculate_oaq=args.calculate_oaq


if not os.path.exists(input_directory):
    print(input_directory)
    print("input directory does not exist. Exiting")
    sys.exit(1)

if not openAIKEY:
    print("OpenAI API key does not exist. Dispute tactics and Argument quality scores will not be computed")    

dialogs = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

print("Building corpus of ", len(dialogs), "dialogs")
print("Calculate politeness: ", politeness_flag)
timestr = time.strftime("%Y%m%d-%H%M%S")


def find_utterance_replied_to(current_speaker,current_utterance_id,text,speakers,utterances_history):
    for person,speaker_value in speakers.items():
        if person != current_speaker and person in text:
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
    return utterances, conv_topic
    
###########################################################################################################################3
aggregate_utterances=[]
collaboration_features_per_dialog=[]
for dialog in dialogs:
    utterances,conv_topic=processDialog(dialog)
    colab=Collaboration(utterances)
    collaboration_features=colab.calculate_collaboration_features() 
    collaboration_features_per_dialog.append(collaboration_features)
    aggregate_utterances=aggregate_utterances+utterances
    

average_collaboration_features_per_dialog=[]
for col_features in collaboration_features_per_dialog:
    dialog_id=list(col_features.keys())[0].split("_")[2]
    col_features_df=pd.DataFrame(col_features)
    col_features_df=col_features_df.transpose()
    mean=col_features_df.mean()
    average_collaboration_features_per_dialog.append({dialog_id:mean.to_dict()})

# Flatten the list of dictionaries
flattened_data = []
for item in average_collaboration_features_per_dialog:
    for key, value in item.items():
        flattened_data.append(value)
df = pd.DataFrame(flattened_data)
mean_values = df.mean()

average_collaboration_features_per_dialog.append({"aggregate_mean":mean_values.to_dict()})

with open('collaboration_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(average_collaboration_features_per_dialog, fout, ensure_ascii=False, indent=4)


with open('collaboration_features_per_utterance_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(collaboration_features_per_dialog, fout, ensure_ascii=False, indent=4)
###########################################################################################################################3
if openAIKEY and calculate_oaq:
    ovargquality_scores_llm_output_dict={}
    for dialog in dialogs:
        utterances,conv_topic=processDialog(dialog)
        dialog_id=utterances[0].id.split("_")[2]
        print("Overall Argument Quality-Proccessing dialog: ",dialog_id," with LLM")
        ovargqual=OAQuality(utterances,conv_topic,openAIKEY,mode)
        ovargument_quality_scores_features=ovargqual.calculate_ovargquality_scores() 
        ovargquality_scores_llm_output_dict[dialog_id]=ovargument_quality_scores_features

    with open('llm_output_oaq'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(ovargquality_scores_llm_output_dict, fout, ensure_ascii=False, indent=4)

    """        
    with open("llm_output_oaq.json", encoding="utf-8") as f:
        oaq = json.load(f)
    ovargquality_scores_llm_output_dict=oaq
    """

    oaq_dim_per_dialog={}
    for dialog_id, turnAnnotations in ovargquality_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label==-1:
                print("LLM output with missing overall argument quality , skipping dialogue\n")
                print(label)
                continue
            parts=label.split("The average overall quality of the arguments presented in the above discussion is:")
            value=parts[1]
            rightParenthesisIndex=value.find("]")
            value=value[0:rightParenthesisIndex]
            value=value.replace("[","").replace("]","")
            oaq_dim_per_dialog[dialog_id]=value

    oaq_mean=np.mean([float(number) for number in list(oaq_dim_per_dialog.values())])
    
    oaq_dim_per_dialog["aggregate_mean"]=str(oaq_mean)



    with open('oaq_per_dialogue_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(oaq_dim_per_dialog, fout, ensure_ascii=False, indent=4)
##########################################################################################################################3
###########################################################################################################################3
if openAIKEY and calculate_maq:
    argqualitydimensions_scores_llm_output_dict={}
    for dialog in dialogs:
        utterances,conv_topic=processDialog(dialog)
        dialog_id=utterances[0].id.split("_")[2]
        print("Argument Quality Dimensions-Proccessing dialog: ",dialog_id," with LLM")
        argqualdimensions=AQualityDimensions(utterances,conv_topic,openAIKEY)
        argument_quality_dimensions_features=argqualdimensions.argquality_dimensions_scores() 
        argqualitydimensions_scores_llm_output_dict[dialog_id]=argument_quality_dimensions_features

    with open('llm_output_maq'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(argqualitydimensions_scores_llm_output_dict, fout, ensure_ascii=False, indent=4)

    """        
    with open("llm_output_maq.json", encoding="utf-8") as f:
        maq = json.load(f)
    argqualitydimensions_scores_llm_output_dict=maq
    """
    arq_dim_per_dialog={}
    for dialog_id, turnAnnotations in argqualitydimensions_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label==-1 or not label.startswith('-Level 1a:'):
                print("LLM output for utterance is ill-formatted, skipping utterance\n")
                print(label)
                continue
            parts=label.split("\n")
            if len(parts)!=14:
                print("LLM output with missing arg quality dimensions, skipping utterance\n")
                print(label)
                continue
            feature={}
            for j in parts:
                entries=j.split(":")
                key=entries[0]
                value=entries[1]
                key=key.replace("-","")
                rightParenthesisIndex=value.find("]")
                value=value[0:rightParenthesisIndex]
                value=value.replace("[","").replace("]","")
                feature[key]=value
            if dialog_id in arq_dim_per_dialog:
                arq_dim_per_dialog[dialog_id].append(feature)
            else:
                arq_dim_per_dialog[dialog_id]=[feature]

    average_argqualdim_per_dialog=[]
    for dialog_id, maq_features in arq_dim_per_dialog.items():
        maq_df=pd.DataFrame(maq_features)
        cols = maq_df.columns
        maq_df[cols] = maq_df[cols].apply(pd.to_numeric, errors='coerce')
        mean=maq_df.mean()
        average_argqualdim_per_dialog.append({dialog_id:mean.to_dict()})

    # Flatten the list of dictionaries
    flattened_data = []
    for item in average_argqualdim_per_dialog:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()
    
    average_argqualdim_per_dialog.append({"aggregate_mean":mean_values.to_dict()})

    with open('maq_mean_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(average_argqualdim_per_dialog, fout, ensure_ascii=False, indent=4)
        
    with open('maq_per_utterance_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(arq_dim_per_dialog, fout, ensure_ascii=False, indent=4)



###########################################################################################################################3

if openAIKEY and calculate_dispute_tactics:
    dispute_tactics_llm_output_dict={}
    for dialog in dialogs:
        utterances,conv_topic=processDialog(dialog)
        dialog_id=utterances[0].id.split("_")[2]
        print("Proccessing dialog: ",dialog_id," with LLM")
        dispute=DisputeTactics(utterances,conv_topic,openAIKEY)
        dispute_features=dispute.calculate_dispute_tectics()
        dispute_tactics_llm_output_dict[dialog_id]=dispute_features
    
    
    with open('llm_dispute_tactics_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(dispute_tactics_llm_output_dict, fout, ensure_ascii=False, indent=4)

    """        
    with open("llm_dispute_tactics_20241123-205834.json", encoding="utf-8") as f:
        d = json.load(f)
    dispute_tactics_llm_output_dict=d
    """
    
    dispute_tactics_per_dialog={}
    for dialog_id, turnAnnotations in dispute_tactics_llm_output_dict.items():
        for label in turnAnnotations:
            if label==-1 or not label.startswith('- Level 0:'):
                print("LLM output for utterance is ill-formatted, skipping utterance\n")
                print(label)
                continue
            parts=label.split("\n")
            feature={}
            for j in parts:
                entries=j.split(":")
                key=entries[0]
                value=entries[1]
                key=key.replace("-","")
                value=value.replace("[","").replace("]","")
                feature[key]=value
            if dialog_id in dispute_tactics_per_dialog:
                dispute_tactics_per_dialog[dialog_id].append(feature)
            else:
                dispute_tactics_per_dialog[dialog_id]=[feature]
    
    average_dispute_tactics_features_per_dialog=[]
    for dialog_id, dt_features in dispute_tactics_per_dialog.items():
        dt_df=pd.DataFrame(dt_features)
        cols = dt_df.columns
        dt_df[cols] = dt_df[cols].apply(pd.to_numeric, errors='coerce')
        mean=dt_df.mean()
        average_dispute_tactics_features_per_dialog.append({dialog_id:mean.to_dict()})
        
    # Flatten the list of dictionaries
    flattened_data = []
    for item in average_dispute_tactics_features_per_dialog:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()
    
    average_dispute_tactics_features_per_dialog.append({"aggregate_mean":mean_values.to_dict()})
        
        
    with open('dispute_tactics_mean_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(average_dispute_tactics_features_per_dialog, fout, ensure_ascii=False, indent=4)
        
    with open('dispute_tactics_features_per_utterance_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(dispute_tactics_per_dialog, fout, ensure_ascii=False, indent=4)
###########################################################################################################################3
if politeness_flag:
    corpus = Corpus(utterances=aggregate_utterances)
    print("Corpus created successfully.")
    
    corpus.print_summary_stats()
    
    # Initialize the TextParser transformer
    parser = TextParser()
    
    # Parse the utterances in the corpus
    corpus = parser.transform(corpus)
    
    # Initialize the PolitenessStrategies transformer
    politeness_transformer = PolitenessStrategies()
    # Transform the corpus to add politeness strategies and markers
    corpus = politeness_transformer.transform(corpus,markers=True)
  
    data=politeness_transformer.summarize(corpus,plot=True)
    data_df = pd.DataFrame([data])
        
    proportions = data_df.sum(axis=0) / len(data_df)
    num_strategies = len(proportions)
    plt.figure(dpi=200, figsize=(24, 24))
    plt.bar(proportions.index, proportions.values)
    plt.xticks(np.arange(0.4, num_strategies + 0.4), rotation=45, ha="right")
    plt.ylabel("% utterance using strategy", size=20)
    plt.yticks(size=15)
    plt.savefig(timestr+'.png')
    
    data_df.to_json('politeness_'+timestr+'.json', orient='records', lines=False,indent=2)
###########################################################################################################################3

import os
import io
import sys
import argparse
import json
from convokit import Corpus, Utterance, Speaker, PolitenessStrategies, TextParser
import pandas as pd
import time
import numpy as np
from collaboration import Collaboration
from dispute_tactics import DisputeTactics
from argument_quality_overall import OAQuality
from argument_quality_per_dimensions import AQualityDimensions
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
                    "--calculate_politeness", default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Whether to calculate politeness or not"
                    )

parser.add_argument(
                    "--openAIKEY", default="",
                    help="Open AI API key"
                    )

parser.add_argument(
                    "--calculate_dispute_tactics", default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Wether to calculate dispute tactics"
                    )


parser.add_argument(
                    "--calculate_maq", default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Whether to calculate multi dimensional argument quality"
                    )


parser.add_argument(
                    "--calculate_oaq", default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Whether to calculate overall argument quality"
                    )


parser.add_argument(
                    "--prompt_mode", default='real',
                    help="mode of prompt for the average overall argument quality score. Permited values: real, rating"
                    )


parser.add_argument(
                    "--include_mod_utterances", default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Whether to include the utterances of the moderator"
                    )


args = parser.parse_args()
input_directory=args.input_directory
politeness_flag=args.calculate_politeness
openAIKEY=args.openAIKEY
calculate_dispute_tactics=args.calculate_dispute_tactics
calculate_maq=args.calculate_maq
calculate_oaq=args.calculate_oaq
mode=args.prompt_mode
moderator_flag=args.include_mod_utterances

if not os.path.exists(input_directory):
    print(input_directory)
    print("input directory does not exist. Exiting")
    sys.exit(1)
    
if not openAIKEY:
    print("OpenAI API key does not exist. Dispute tactics and Argument quality scores will not be computed.")


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
                  if ((isinstance(log[1], str)) and (len(log[1]) > MESSAGE_THREASHOLD_IN_CHARS))
                 ]
    
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
###########################################################################################################################3
aggregate_utterances=[]
collaboration_features_per_disc=[]
for disc in discussions:
    utterances,conv_topic=processDiscussion(disc)
    colab=Collaboration(utterances)
    collaboration_features=colab.calculate_collaboration_features() 
    collaboration_features_per_disc.append(collaboration_features)
    aggregate_utterances=aggregate_utterances+utterances
    

average_collaboration_features_per_disc=[]
for col_features in collaboration_features_per_disc:
    disc_id=list(col_features.keys())[0].split("_")[2]
    col_features_df=pd.DataFrame(col_features)
    col_features_df=col_features_df.transpose()
    mean=col_features_df.mean()
    average_collaboration_features_per_disc.append({disc_id:mean.to_dict()})


flattened_data = []
for item in average_collaboration_features_per_disc:
    for key, value in item.items():
        flattened_data.append(value)
df = pd.DataFrame(flattened_data)
mean_values = df.mean()

average_collaboration_features_per_disc.append({"aggregate_mean":mean_values.to_dict()})

with open('collaboration_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(average_collaboration_features_per_disc, fout, ensure_ascii=False, indent=4)


with open('collaboration_features_per_utterance_'+timestr+'.json', 'w',encoding="utf-8") as fout:
    json.dump(collaboration_features_per_disc, fout, ensure_ascii=False, indent=4)
###########################################################################################################################3
if openAIKEY and calculate_oaq:
    ovargquality_scores_llm_output_dict={}
    for disc in discussions:
        utterances,conv_topic=processDiscussion(disc)
        disc_id=utterances[0].id.split("_")[2]
        print("Overall Argument Quality-Proccessing discussion: ",disc_id," with LLM")
        ovargqual=OAQuality(utterances,conv_topic,openAIKEY,mode)
        ovargument_quality_scores_features=ovargqual.calculate_ovargquality_scores() 
        ovargquality_scores_llm_output_dict[disc_id]=ovargument_quality_scores_features

    with open('llm_output_oaq'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(ovargquality_scores_llm_output_dict, fout, ensure_ascii=False, indent=4)

    """        
    with open("llm_output_oaq.json", encoding="utf-8") as f:
        oaq = json.load(f)
    ovargquality_scores_llm_output_dict=oaq
    """

    oaq_dim_per_disc={}
    for disc_id, turnAnnotations in ovargquality_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label==-1:
                print("LLM output with missing overall argument quality , skipping discussion\n")
                print(label)
                continue
            parts=label.split("The average overall quality of the arguments presented in the above discussion is:")
            value=parts[1]
            rightParenthesisIndex=value.find("]")
            leftParenthesisIndex=value.find("[")
            value=value[leftParenthesisIndex:rightParenthesisIndex]
            value=value.replace("[","").replace("]","")            
            oaq_dim_per_disc[disc_id]=value

    oaq_mean=np.mean([float(number) for number in list(oaq_dim_per_disc.values())])
    
    oaq_dim_per_disc["aggregate_mean"]=str(oaq_mean)


    with open('oaq_per_disc_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(oaq_dim_per_disc, fout, ensure_ascii=False, indent=4)
##########################################################################################################################3
###########################################################################################################################3
if openAIKEY and calculate_maq:
    argqualitydimensions_scores_llm_output_dict={}
    for disc in discussions:
        utterances,conv_topic=processDiscussion(disc)
        disc_id=utterances[0].id.split("_")[2]
        print("Argument Quality Dimensions-Proccessing disc: ",disc_id," with LLM")
        argqualdimensions=AQualityDimensions(utterances,conv_topic,openAIKEY)
        argument_quality_dimensions_features=argqualdimensions.argquality_dimensions_scores() 
        argqualitydimensions_scores_llm_output_dict[disc_id]=argument_quality_dimensions_features

    with open('llm_output_maq'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(argqualitydimensions_scores_llm_output_dict, fout, ensure_ascii=False, indent=4)

    """        
    with open("llm_output_maq.json", encoding="utf-8") as f:
        maq = json.load(f)
    argqualitydimensions_scores_llm_output_dict=maq
    """
    arq_dim_per_disc={}
    for disc_id, turnAnnotations in argqualitydimensions_scores_llm_output_dict.items():
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
                leftParenthesisIndex=value.find("[")
                value=value[leftParenthesisIndex:rightParenthesisIndex]
                value=value.replace("[","").replace("]","")            
                feature[key]=value
            if disc_id in arq_dim_per_disc:
                arq_dim_per_disc[disc_id].append(feature)
            else:
                arq_dim_per_disc[disc_id]=[feature]

    average_argqualdim_per_disc=[]
    for disc_id, maq_features in arq_dim_per_disc.items():
        maq_df=pd.DataFrame(maq_features)
        cols = maq_df.columns
        maq_df[cols] = maq_df[cols].apply(pd.to_numeric, errors='coerce')
        mean=maq_df.mean()
        average_argqualdim_per_disc.append({disc_id:mean.to_dict()})


    flattened_data = []
    for item in average_argqualdim_per_disc:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()
    
    average_argqualdim_per_disc.append({"aggregate_mean":mean_values.to_dict()})

    with open('maq_mean_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(average_argqualdim_per_disc, fout, ensure_ascii=False, indent=4)
        
    with open('maq_per_utterance_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(arq_dim_per_disc, fout, ensure_ascii=False, indent=4)
###########################################################################################################################3
if openAIKEY and calculate_dispute_tactics:
    dispute_tactics_llm_output_dict={}
    for disc in discussions:
        utterances,conv_topic=processDiscussion(disc)
        disc_id=utterances[0].id.split("_")[2]
        print("Proccessing disc: ",disc_id," with LLM")
        dispute=DisputeTactics(utterances,conv_topic,openAIKEY)
        dispute_features=dispute.calculate_dispute_tectics()
        dispute_tactics_llm_output_dict[disc_id]=dispute_features
    
    
    with open('llm_dispute_tactics_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(dispute_tactics_llm_output_dict, fout, ensure_ascii=False, indent=4)

    """        
    with open("llm_dispute_tactics_20241123-205834.json", encoding="utf-8") as f:
        d = json.load(f)
    dispute_tactics_llm_output_dict=d
    """
    
    dispute_tactics_per_disc={}
    for disc_id, turnAnnotations in dispute_tactics_llm_output_dict.items():
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
            if disc_id in dispute_tactics_per_disc:
                dispute_tactics_per_disc[disc_id].append(feature)
            else:
                dispute_tactics_per_disc[disc_id]=[feature]
    
    average_dispute_tactics_features_per_disc=[]
    for disc_id, dt_features in dispute_tactics_per_disc.items():
        dt_df=pd.DataFrame(dt_features)
        cols = dt_df.columns
        dt_df[cols] = dt_df[cols].apply(pd.to_numeric, errors='coerce')
        mean=dt_df.mean()
        average_dispute_tactics_features_per_disc.append({disc_id:mean.to_dict()})
        

    flattened_data = []
    for item in average_dispute_tactics_features_per_disc:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()
    
    average_dispute_tactics_features_per_disc.append({"aggregate_mean":mean_values.to_dict()})
        
        
    with open('dispute_tactics_mean_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(average_dispute_tactics_features_per_disc, fout, ensure_ascii=False, indent=4)
        
    with open('dispute_tactics_features_per_utterance_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(dispute_tactics_per_disc, fout, ensure_ascii=False, indent=4)
###########################################################################################################################3
politeness_per_disc={}
if politeness_flag:
    for disc in discussions:
        utterances,conv_topic=processDiscussion(disc)
        disc_id=utterances[0].id.split("_")[2]
        corpus = Corpus(utterances=utterances)
        print("Corpus created successfully.")
        parser = TextParser()
        corpus = parser.transform(corpus)
        politeness_transformer = PolitenessStrategies()
        corpus = politeness_transformer.transform(corpus,markers=True)
        data=politeness_transformer.summarize(corpus,plot=False)
        politeness_per_disc[disc_id]=dict(data)
       
    average_politeness=[]
    for disc_id, plt_features in politeness_per_disc.items():
        plt_df=pd.DataFrame(plt_features,index=[0])
        cols = plt_df.columns
        plt_df[cols] = plt_df[cols].apply(pd.to_numeric, errors='coerce')
        mean=plt_df.mean()
        average_politeness.append({disc_id:mean.to_dict()})
           
    flattened_data = []
    for item in average_politeness:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()
    
    average_politeness.append({"aggregate_mean":mean_values.to_dict()})
    
    with open('average_politeness_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(average_politeness, fout, ensure_ascii=False, indent=4)
    """    
    with open('politeness_per_disc_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(politeness_per_disc, fout, ensure_ascii=False, indent=4)
    """
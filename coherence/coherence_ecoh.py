import os
import sys
import argparse
import json
from tqdm import tqdm
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
#############################################################

base_model = "Qwen/Qwen1.5-4B-Chat"
adapter_model = "Johndfm/ECoh-4B"
#############################################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_directory", default="input_directory_with_dialogs",
    help="The folder with the llm dialogs to process")
args = parser.parse_args()

input_directory=args.input_directory

if not os.path.exists(input_directory):
    print(input_directory)
    print("input directory does not exist. Exiting")
    sys.exit(1)

dialogs = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
print("Building corpus of ", len(dialogs), "dialogs")
timestr = time.strftime("%Y%m%d-%H%M%S")
#############################################################
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, adapter_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
#############################################################
device="cpu"
model = model.to(device)


def inferenceModel(history,response):
    
    textForInference =f"Context:\n {history} \n\nResponse:\n {response}"
    
    messages = [
          {"role": "system", "content": "You are a Coherence evaluator."},
          {"role": "user", "content": f"{textForInference}\n\nGiven the context, is the response Coherent (Yes/No)? Explain your reasoning."}
               ]
    
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
    )
    
    generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def processDialog(dialog):
    with open(dialog, 'r', encoding="utf-8") as file:
        data = json.load(file)
    
    conversation_id=data["id"]
    has_moderator=True
    if not data["moderator"]:
        has_moderator=False
    
    speakers = {
            user:user_type for user, user_type in zip(data['users'], data['user_types'])
           }
    
    if has_moderator:
        speakers["moderator"]=data["moderator_type"]
        
    utterances = []
    data['logs']=[(log[0],log[1]) for log in data['logs'] if len(log[1])>25]
    for i, log in enumerate(data['logs']):
        speaker, text = log
        text=text.replace('\r\n',' ').replace('\n',' ').strip()
        utterances.append((text,speaker,f"conv_{conversation_id}_utt_{i}"))
   
    context_list=[]
    response_list=[]
    previous_context=""
    seperator=" \n "
    context=""
    for i,u in enumerate(utterances):
        text,speaker,utt_id=u
        if (i+1>=len(utterances)):
            break
        text= speaker+": "+text
        if(i==0):
            context=text
            previous_context=text
            response_speaker=utterances[i+1][1]
            response=response_speaker+": "+utterances[i+1][0]
        else:
            context=previous_context+seperator+" "+text
            response_speaker=utterances[i+1][1]
            response=response_speaker+": "+utterances[i+1][0]
            previous_context=context
        context_list.append(context)
        response_list.append(response)
    
    inference_result_list=[]
    for cont,resp in tqdm(zip(context_list,response_list),total=len(response_list)):
        result=inferenceModel(cont,resp)
        inference_result_list.append(result)
    return inference_result_list,context_list,response_list, conversation_id
    
    
aggregate_results={}
aggregate_context={}
aggregate_response={}
for dialog in dialogs:
    results,context_list,response_list, conversation_id = processDialog(dialog)
    aggregate_results[conversation_id]=results
    aggregate_context[conversation_id]=context_list
    aggregate_response[conversation_id]=response_list

booleans={}
reasons={}
for dialog_id, value in aggregate_results.items():
    for res in value:
        parts=res.split("The answer is")
        reason=parts[0]
        res_str=parts[1]
        boolean_res=0
        if "Yes" in res_str:
            boolean_res=1
        if dialog_id in booleans:
            booleans[dialog_id].append(boolean_res)
            reasons[dialog_id].append(reason)
        else:
            booleans[dialog_id]=[boolean_res]
            reasons[dialog_id]=[reason]

with open('dialogue_turnlevel_agresults_'+timestr+'.json', 'w',encoding="utf-8") as fout:
      json.dump(aggregate_results, fout, ensure_ascii=False, indent=4) 

with open('dialogue_turnlevel_reasons_'+timestr+'.json', 'w',encoding="utf-8") as fout:
      json.dump(reasons, fout, ensure_ascii=False, indent=4) 

with open('dialogue_turnlevel_boolean_results_'+timestr+'.json', 'w',encoding="utf-8") as fout:
      json.dump(booleans, fout, ensure_ascii=False, indent=4) 


average_coherence_per_dialog=[]
for dialog_id, score_list in booleans.items():
    utt_coh_per_dialog_df=pd.DataFrame(score_list)
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

with open('dialogue_echo_results_'+timestr+'.json', 'w',encoding="utf-8") as fout:
      json.dump(average_coherence_per_dialog, fout, ensure_ascii=False, indent=4) 

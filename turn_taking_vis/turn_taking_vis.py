import os
import sys
import argparse
import json
from convokit import Corpus, Utterance, Speaker, Coordination
import time
import matplotlib.pyplot as plt
import io

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

def save_stdout_to_image(func, conversation_id,  *args, **kwargs):
    # Capture the standard output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()  
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    # Get the output as a string
    output = buffer.getvalue()  
    # Create an image from the text
    fig, ax = plt.subplots()
    ax.text(0, 1, output, fontsize=12, ha='left', va='top', wrap=True, transform=ax.transAxes)
    ax.axis('off')
    # Save the image
    folder_path='output_images/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path+conversation_id+'.png', bbox_inches='tight')

for dialog in dialogs:
    utterances,speakers,data,utterances_history,data_cord,conversation_id,has_moderator=processDialog(dialog)
    corpus = Corpus(utterances=utterances)
    print("Corpus created successfully.")
    corpus.print_summary_stats()
    coord = Coordination()
    coord.fit(corpus)
    coord.transform(corpus)
    conv0=corpus.get_conversation(conversation_id)
    save_stdout_to_image(conv0.print_conversation_structure,conversation_id)
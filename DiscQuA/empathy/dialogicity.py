# -*- coding: utf-8 -*-

import json
import os
import sys
import time

import openai
import pandas as pd
from convokit import Speaker, Utterance

ini = """Below is a set of communicative functions (presented in Macagno et al., 2022) of utterances in human discussions aimed in capturing their true intention in terms of dialogicity (potential other-orientedness).
Generally speaking, an utterance is considered less dialogic when it does not result in a continuation of the discussion and does not build on the previous discussion discourse or does not explore a viewpoint. 
In contrast, an utterance is considered more dialogic when it considers others' discussants viewpoints."""


dialogical_labels = """\n\n
Dialogical labels:

Label 0: Stating - An utterance that is relevant to the discussion topic and conveys information, viewpoints or value judgments on a state of affairs or another viewpoint without providing any reason. Note that this function can be performed by sentences that are not necessarily assertive (e.g., interrogative).

Label 1: Accepting/Discarding - An utterance that is relevant to the discussion topic and accepts or acknowledges or challenges or rejects an opinion expressed by another speaker, without providing further reasons or addressing problematic background information.

Label 2: Managerial - An utterance that is relevant to the discussion topic and establishes the norms of the discussion. Examples are utterances aiming to coordinate turn taking in a discussion.

Label 3a: Expanding/Low relevance - An utterance that tries to make one's own position more acceptable or understandable to the discussants, without considering the previous utterances. 

Label 3b: Expanding/High relevance - An utterance that tries to undestand and interact with another intelocutors position or make one's own position more acceptable or understandable to the discussants, while it considers the previous utterances. 
    
Label 4a: Metadialogical/Low relevance - An utterance that addresses the pragmatic, linguistic or content dimensions of discussion units unrelated to the previous utterances. Also utterances that reference the discussion process without the intention of discussing its goals.

Label 4b: Metadialogical/High relevance - An utterance that addresses the pragmatic, linguistic or content dimensions of discussion units of the previous comments. Examples are utterances addressing previous utterances and requesting meaning explanation or undestanding confirmation or expressing lack of understanding or explaining the meaning of expressions. Also utterances that make a connection between the current state of the discussion and its supposed goal.  

Label 5: Reasoning - An utterance that (a) provides justification for a change of a position, (b) advances reasons against other's arguments or positions, (c) synthesizes or compares ideas, or generalizes, (d) makes reasoning explicit by providing explanations, justifications, argumentation, analogies, or evidence.

Label 6: Metadialogical Reasoning - An utterance that satisfies the conditions of labels 4a or 4b (Metadialogical Low/High relevance) and 5 (Reasoning). Examples are utterances that attack viewpoints or arguments by focusing on the meaning of their expressions or the implicit premises that are presupposed.

Label 7a: Inviting/Low relevance - An utterance that invites others to express their viewpoints, or what they think about a certain interpretation, without explicitly stating speaker's interest in better understanding the other's opinion.

Label 7b: Inviting/High relevance - An utterance that invites others to express their viewpoints, or what they think about a certain interpretation, by requesting explanations, clarifications or justifications of a previous contribution.

Label 8: Other - For utterances not covered by the above labels.

"""
final = """\n\n
Given a discussion history (which can be empty if the new utterance is the first utterance made in the discussion), please analyze a new utterance from a user (identified by a unique user_id) in a conversation discussing with others about this potentially controversial post.
Post: {post} 
Each utterance is a brief Reddit-based comment responding to the post and other users' comments on it.
    
*CONVERSATION HISTORY*: "{conv_history}"

*NEW UTTERANCE*: "{utterance}"

Noteworthy, the discussion history is provided for you to simply understand the utterances made before the new utterance so as to help you better annotate the new utterance.

Thus, please do not annotate the entire discussion but annotate only the new utterance by determining its appropriate dialogical label(s). Please provide the final answer directly with no reasoning steps.

If the new utterance fits multiple categories, list all that apply. Ensure that your final answer clearly identifies whether each of the dialogical labels (0-8) are applicable (1) or not applicable (0) to the new utterance.

For clarity, your response should succinctly be presented in a structured list format, indicating the presence (1) or absence (0) of label as follows:

- Label 0: [1/0]
- Label 1: [1/0]
- Label 2: [1/0]
- Label 3a: [1/0]
- Label 3b: [1/0]
- Label 4a: [1/0]
- Label 4b: [1/0]
- Label 5: [1/0]
- Label 6: [1/0]
- Label 7a: [1/0]
- Label 7b: [1/0]
- Label 8: [1/0]
"""

MODERATOR = "moderator"
MESSAGE_THREASHOLD_IN_CHARS = 25
HARDCODED_MODEL = "hardcoded"


def processDiscussion(discussion, moderator_flag):
    with open(discussion, "r", encoding="utf-8") as file:
        data = json.load(file)

    conversation_id = data["id"]
    speakers = {user: Speaker(id=user) for user in data["users"]}

    if not moderator_flag:
        data["logs"] = [
            (log[0], log[1], log[2], log[3])
            for log in data["logs"]
            if log[0] != MODERATOR
        ]

    utterances = []

    data["logs"] = [
        (log[0], log[1], log[2], log[3])
        for log in data["logs"]
        if ((isinstance(log[1], str)) and (len(log[1]) > MESSAGE_THREASHOLD_IN_CHARS))
    ]

    for i, log in enumerate(data["logs"]):
        speaker, text, model, message_id = log
        text = text.rstrip().lstrip()
        if i == 0 and model == HARDCODED_MODEL:
            conv_topic = text
        utterances.append(
            Utterance(
                id=f"utt_{i}_{conversation_id}",
                speaker=speakers[speaker],
                conversation_id=str(conversation_id),
                text=text,
                meta={"timestamp": data["timestamp"]},
            )
        )
    return utterances, conv_topic


def prompt_gpt4(prompt, key):
    openai.api_key = key

    ok = False
    counter = 0
    while (
        not ok
    ):  # to avoid "ServiceUnavailableError: The server is overloaded or not ready yet."
        counter = counter + 1
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0,
            )
            ok = True
        except Exception as ex:
            print("error", ex)
            print("sleep for 5 seconds")
            time.sleep(5)
            if counter > 10:
                return -1
    return response["choices"][0]["message"]["content"]


def calculate_dialog_labels(utts, topic, openAIKEY):
    prompt = ini + dialogical_labels + final
    conv_hist = ""
    annotations_ci = []
    for index, utt in enumerate(utts):
        text = utt.text
        speaker = utt.get_speaker().id
        if index == 0:
            conv_hist = ""
        else:
            conv_hist = (
                conv_hist
                + "\n"
                + "<user_name="
                + utts[index - 1].get_speaker().id
                + "\n"
                + utts[index - 1].text
                + "\n"
            )
        try:
            formatted_prompt = prompt.format(
                utterance="<user_name=" + speaker + ">" + "\n" + text,
                conv_history=conv_hist,
                post=topic,
            )
            response_text = prompt_gpt4(formatted_prompt, openAIKEY)
            # print(formatted_prompt)
            annotations_ci.append(response_text)
        except Exception as e:
            print("Error: ", e)
            annotations_ci.append(-1)
    return annotations_ci


def dialogicity(input_directory, openAIKEY, moderator_flag=True):

    if not os.path.exists(input_directory):
        print(input_directory)
        print("input directory does not exist. Exiting")
        sys.exit(1)

    if not openAIKEY:
        print(
            "OpenAI API key does not exist. Dialogical labels will not be computed. Exiting"
        )
        sys.exit(1)

    discussions = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if os.path.isfile(os.path.join(input_directory, f))
    ]
    print("Building corpus of ", len(discussions), "discussions")
    timestr = time.strftime("%Y%m%d-%H%M%S")

    dialog_labels_llm_output_dict = {}
    for disc in discussions:
        try:
            utterances, conv_topic = processDiscussion(disc, moderator_flag)
            disc_id = utterances[0].id.split("_")[2]
            print(
                "Dialogical labels Per Response-Proccessing discussion: ",
                disc_id,
                " with LLM",
            )
            dial_labels_per_resp = calculate_dialog_labels(
                utterances, conv_topic, openAIKEY
            )
            dialog_labels_llm_output_dict[disc_id] = dial_labels_per_resp
            print("Sleeping for 60 seconds, for openAI quota")
            time.sleep(60)
        except Exception as e:
            print("Error: ", e)
            print(disc)

    with open(
        "llm_output_diallabels_per_response_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(dialog_labels_llm_output_dict, fout, ensure_ascii=False, indent=4)

    """        
    with open("llm_output_diallabels_per_response_.json", encoding="utf-8") as f:
        dial_labels = json.load(f)
    dialog_labels_llm_output_dict=dial_labels
    """

    dialo_labels_per_response = {}
    for disc_id, turnAnnotations in dialog_labels_llm_output_dict.items():
        counter = 0
        for label in turnAnnotations:
            if label == -1:
                print("LLM output with missing dialogicity label, skipping response\n")
                print(label)
                counter += 1
                continue
            parts = label.split("\n")
            feature = {}
            for j in parts:
                entries = j.split(":")
                key = entries[0]
                value = entries[1]
                key = key.replace("-", "")
                value = value.replace("[", "").replace("]", "")
                feature[key] = value
            if disc_id in dialo_labels_per_response:
                dialo_labels_per_response[disc_id].append(feature)
            else:
                dialo_labels_per_response[disc_id] = [feature]

    average_diallabel_per_disc = []
    for disc_id, dial_label in dialo_labels_per_response.items():
        dial_label_per_disc_df = pd.DataFrame(dial_label)

        cols = dial_label_per_disc_df.columns
        dial_label_per_disc_df[cols] = dial_label_per_disc_df[cols].apply(
            pd.to_numeric, errors="coerce"
        )
        mean = dial_label_per_disc_df.mean()
        average_diallabel_per_disc.append({disc_id: mean.to_dict()})

    flattened_data = []
    for item in average_diallabel_per_disc:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()

    average_diallabel_per_disc.append({"aggregate_mean": mean_values.to_dict()})

    with open(
        "disc_aver_diallabel_per_response_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(average_diallabel_per_disc, fout, ensure_ascii=False, indent=4)

    with open(
        "diallabel_per_response_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(dialo_labels_per_response, fout, ensure_ascii=False, indent=4)

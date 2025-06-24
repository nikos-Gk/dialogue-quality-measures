# -*- coding: utf-8 -*-
import time

from DiscQuA.utils import (
    getModel,
    getUtterances,
    prompt_gpt4,
    save_dict_2_json,
    sleep,
    validateInputParams,
)

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


def calculate_dialog_labels(utts, topic, openAIKEY, model_type, model, ctx):
    prompt = ini + dialogical_labels + final
    annotations_ci = []
    #
    for index, utt in enumerate(utts):
        conv_hist = ""
        text = utt.text
        speaker = utt.get_speaker().id
        if index > 0:
            start_ctx = max(0, index - ctx)
            for i in range(start_ctx, index):
                prev_utt = utts[i]
                prev_speaker = prev_utt.get_speaker().id
                prev_text = prev_utt.text
                conv_hist += f"\n<user_name={prev_speaker}>\n{prev_text}\n"

        try:
            formatted_prompt = prompt.format(
                utterance="<user_name=" + speaker + ">" + "\n" + text,
                conv_history=conv_hist,
                post=topic,
            )
            response_text = prompt_gpt4(formatted_prompt, openAIKEY, model_type, model)
            # print(formatted_prompt)
            annotations_ci.append(response_text)

        except Exception as e:
            print("Error: ", e)
            annotations_ci.append(-1)
    return annotations_ci


def dialogicity(
    message_list,
    speakers_list,
    disc_id,
    openAIKEY,
    model_type="openai",
    model_path="",
    gpu=False,
    ctx=1,
):

    validateInputParams(model_type, openAIKEY, model_path)
    print("Building corpus of ", len(message_list), "utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None
    if model_type == "llama":
        llm = getModel(model_path, gpu)

    dialog_labels_llm_output_dict = {}

    try:
        utterances, speakers = getUtterances(
            message_list, speakers_list, disc_id, replyto_list=[]
        )
        conv_topic = message_list[0]
        print(
            "Dialogical labels Per Response-Proccessing discussion: ",
            disc_id,
            " with LLM",
        )

        dial_labels_per_resp = calculate_dialog_labels(
            utterances, conv_topic, openAIKEY, model_type, llm, ctx
        )
        dialog_labels_llm_output_dict[disc_id] = dial_labels_per_resp
        sleep(model_type)
    except Exception as e:
        print("Error: ", e)
        print(disc_id)

    save_dict_2_json(
        dialog_labels_llm_output_dict,
        "llm_output_diallabels_per_response",
        disc_id,
        timestr,
    )

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
            try:
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
            except Exception as e:
                print(e)

    save_dict_2_json(
        dialo_labels_per_response, "diallabel_per_response", disc_id, timestr
    )
    return dialo_labels_per_response

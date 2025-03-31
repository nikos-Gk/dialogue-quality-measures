# -*- coding: utf-8 -*-
import json
import os
import sys
import time

import openai
import pandas as pd
from convokit import Speaker, Utterance

prompt = """\n\n
You will be presented with a conversation history (which can be empty if the new utterance is the first utterance made in the conversation) from a 
discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating inflammatory and aggressive speech.

*CONVERSATION HISTORY*: "{conv_history}"

*RESPONSE*: "{response}"

You are a toxicity evaluator. 
Given the post that the discussion is based on and the conversation history, you have to assign a label that indicates the toxicity of the response.
The possible labels are 1: Not Toxic, 2: Barely Toxic, 3: Almost Toxic, 4: Toxic and 5: Extremely Toxic.
Noteworthy, the conversation history is provided for you to simply understand the utterances made before the new utterance so as to help you better annotate the toxicity of the new response.
Please provide the final answer directly with no reasoning steps.
For clarity, your evaluation should be presented with the statement: 'The toxicity of the new response is: [X]', where X is the label you've determined. 
Please, ensure that your last statement is the label in brackets [].
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
            #            time.sleep(5)
            if counter > 10:
                return -1
    return response["choices"][0]["message"]["content"]


def calculate_response_toxicity_score(utts, topic, openAIKEY):
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
                response="<user_name=" + speaker + ">" + "\n" + text,
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


def toxicity(input_directory, openAIKEY, moderator_flag=True):

    if not os.path.exists(input_directory):
        print(input_directory)
        print("input directory does not exist. Exiting")
        sys.exit(1)

    if not openAIKEY:
        print(
            "OpenAI API key does not exist. Toxicity labels will not be computed. Exiting"
        )
        sys.exit(1)

    discussions = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if os.path.isfile(os.path.join(input_directory, f))
    ]
    print("Building corpus of ", len(discussions), "discussions")
    timestr = time.strftime("%Y%m%d-%H%M%S")

    tox_per_resp_scores_llm_output_dict = {}
    for disc in discussions:
        try:
            utterances, conv_topic = processDiscussion(disc, moderator_flag)
            disc_id = utterances[0].id.split("_")[2]
            print(
                "Toxicilty Label Per Response-Proccessing discussion: ",
                disc_id,
                " with LLM",
            )
            tox_per_resp = calculate_response_toxicity_score(
                utterances, conv_topic, openAIKEY
            )
            tox_per_resp_scores_llm_output_dict[disc_id] = tox_per_resp
            print("Sleeping for 60 seconds, for openAI quota")
            time.sleep(60)
        except Exception as e:
            print("Error: ", e)
            print(disc)

    with open(
        "llm_output_tox_per_response_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(
            tox_per_resp_scores_llm_output_dict, fout, ensure_ascii=False, indent=4
        )

    """            
    with open("llm_output_tox_per_response_.json", encoding="utf-8") as f:
        tox_scores = json.load(f)
    tox_per_resp_scores_llm_output_dict=tox_scores
    """

    toxicity_scores_per_response = {}
    for disc_id, turnAnnotations in tox_per_resp_scores_llm_output_dict.items():
        counter = 0
        ut_dict = {}
        for label in turnAnnotations:
            if label == -1:
                print(
                    "LLM output with missing toxicity response label , skipping response\n"
                )
                print(label)
                counter += 1
                continue
            parts = label.split("toxicity of the new response is:")
            value = parts[1]
            rightParenthesisIndex = value.find("]")
            leftParenthesisIndex = value.find("[")
            value = value[leftParenthesisIndex:rightParenthesisIndex]
            value = value.replace("[", "").replace("]", "")
            ut_dict[disc_id + "_" + str(counter)] = value
            counter += 1
        toxicity_scores_per_response[disc_id] = ut_dict

    average_toxicity_per_disc = []
    for disc_id, utt_coh in toxicity_scores_per_response.items():
        utt_coh_per_disc_df = pd.DataFrame(utt_coh, index=[0])
        utt_coh_per_disc_df = utt_coh_per_disc_df.transpose()
        cols = utt_coh_per_disc_df.columns
        utt_coh_per_disc_df[cols] = utt_coh_per_disc_df[cols].apply(
            pd.to_numeric, errors="coerce"
        )
        mean = utt_coh_per_disc_df.mean()
        average_toxicity_per_disc.append({disc_id: mean.to_dict()[0]})

    flattened_data = []
    for item in average_toxicity_per_disc:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()

    average_toxicity_per_disc.append({"aggregate_mean": mean_values.to_dict()[0]})

    with open(
        "disc_aver_tox_per_response_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(average_toxicity_per_disc, fout, ensure_ascii=False, indent=4)

    with open(
        "toxicity_per_response_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(toxicity_scores_per_response, fout, ensure_ascii=False, indent=4)

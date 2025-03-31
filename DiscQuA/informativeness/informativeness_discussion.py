# -*- coding: utf-8 -*-
import json
import os
import sys
import time

import numpy as np
import openai
from convokit import Speaker, Utterance

#################################################################################
# sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
# sys.stderr=io.TextIOWrapper(sys.stderr.buffer,encoding="utf-8")
#################################################################################
prompt = """{conv_text}'\n\n\n

The texts above show a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating 
inflammatory and aggressive speech.
Now, please use chain-of-thought reasoning to rate the informativeness of the above discussion.
After the Chain-of-Thoughts reasoning steps, rate the informativeness of the entire discussion on a scale of 1 to 5, where 1 is of poor information quality (uninformative) and 5 of high information quality (informative). 
Conclude your evaluation with the statement: 'The informativeness of the comments presented in the above discussion is: [X]', where X is the rating you've determined. 
Please, ensure that your last statement is the score in brackets [].
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


def calculate_discussion_informativeness_score(utts, topic, key):
    conv_text = ""
    for utt in utts:
        text = utt.text
        speaker = utt.get_speaker().id
        conv_text = conv_text + speaker + ": " + text + "\n\n"
        formatted_prompt = prompt.format(conv_text=conv_text, post=topic)
    annotations_ci = []
    try:
        response_text = prompt_gpt4(formatted_prompt, key)
        # print(formatted_prompt)
        annotations_ci.append(response_text)
    except Exception as e:
        print("Error: ", e)
        annotations_ci.append(-1)
    return annotations_ci


def informativeness(input_directory, openAIKEY, moderator_flag=True):

    if not os.path.exists(input_directory):
        print(input_directory)
        print("input directory does not exist. Exiting")
        sys.exit(1)

    if not openAIKEY:
        print(
            "OpenAI API key does not exist. Informativeness scores will not be computed. Exiting"
        )
        sys.exit(1)

    discussions = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if os.path.isfile(os.path.join(input_directory, f))
    ]
    print("Building corpus of ", len(discussions), "discussions")
    timestr = time.strftime("%Y%m%d-%H%M%S")

    inform_scores_llm_output_dict = {}
    for disc in discussions:
        utterances, conv_topic = processDiscussion(disc, moderator_flag)
        disc_id = utterances[0].id.split("_")[2]
        print(
            "Overall Informativeness Score-Proccessing discussion: ",
            disc_id,
            " with LLM",
        )
        inform_score = calculate_discussion_informativeness_score(
            utterances, conv_topic, openAIKEY
        )
        inform_scores_llm_output_dict[disc_id] = inform_score

    with open(
        "llm_output_informativeness_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(inform_scores_llm_output_dict, fout, ensure_ascii=False, indent=4)

    """        
    with open("llm_output_informativeness_.json", encoding="utf-8") as f:
        info_scores = json.load(f)
    inform_scores_llm_output_dict=info_scores
    """

    inform_scores_per_disc = {}
    for disc_id, turnAnnotations in inform_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label == -1:
                print(
                    "LLM output with missing overall informativeness score , skipping discussion\n"
                )
                print(label)
                continue
            parts = label.split(
                "informativeness of the comments presented in the above discussion is:"
            )
            value = parts[1]
            rightParenthesisIndex = value.find("]")
            leftParenthesisIndex = value.find("[")
            value = value[leftParenthesisIndex:rightParenthesisIndex]
            value = value.replace("[", "").replace("]", "")
            inform_scores_per_disc[disc_id] = value

    inform_scores_mean = np.mean(
        [float(number) for number in list(inform_scores_per_disc.values())]
    )

    inform_scores_per_disc["aggregate_mean"] = str(inform_scores_mean)

    with open(
        "inform_scores_per_discussion_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(inform_scores_per_disc, fout, ensure_ascii=False, indent=4)

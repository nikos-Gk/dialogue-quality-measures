import json
import os
import sys
import time

import numpy as np
from convokit import Speaker, Utterance

from .argument_quality_overall import OAQuality

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


def calculate_overall_arg_quality(
    input_directory, openAIKEY, mode="real", moderator_flag=True
):
    if not os.path.exists(input_directory):
        print(input_directory)
        print("input directory does not exist. Exiting")
        sys.exit(1)

    if not openAIKEY:
        print(
            "OpenAI API key does not exist. Overall argument quality scores will not be computed.Exiting."
        )
        sys.exit(1)

    discussions = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if os.path.isfile(os.path.join(input_directory, f))
    ]
    print("Building corpus of ", len(discussions), "discussions")
    timestr = time.strftime("%Y%m%d-%H%M%S")

    ovargquality_scores_llm_output_dict = {}
    for disc in discussions:
        utterances, conv_topic = processDiscussion(disc, moderator_flag)
        disc_id = utterances[0].id.split("_")[2]
        print("Overall Argument Quality-Proccessing discussion: ", disc_id, " with LLM")
        ovargqual = OAQuality(utterances, conv_topic, openAIKEY, mode)
        ovargument_quality_scores_features = ovargqual.calculate_ovargquality_scores()
        ovargquality_scores_llm_output_dict[disc_id] = (
            ovargument_quality_scores_features
        )

    with open("llm_output_oaq" + timestr + ".json", "w", encoding="utf-8") as fout:
        json.dump(
            ovargquality_scores_llm_output_dict, fout, ensure_ascii=False, indent=4
        )

    """        
    with open("llm_output_oaq.json", encoding="utf-8") as f:
        oaq = json.load(f)
    ovargquality_scores_llm_output_dict=oaq
    """

    oaq_dim_per_disc = {}
    for disc_id, turnAnnotations in ovargquality_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label == -1:
                print(
                    "LLM output with missing overall argument quality , skipping discussion\n"
                )
                print(label)
                continue
            parts = label.split(
                "The average overall quality of the arguments presented in the above discussion is:"
            )
            value = parts[1]
            rightParenthesisIndex = value.find("]")
            leftParenthesisIndex = value.find("[")
            value = value[leftParenthesisIndex:rightParenthesisIndex]
            value = value.replace("[", "").replace("]", "")
            oaq_dim_per_disc[disc_id] = value

    oaq_mean = np.mean([float(number) for number in list(oaq_dim_per_disc.values())])

    oaq_dim_per_disc["aggregate_mean"] = str(oaq_mean)

    with open("oaq_per_disc_" + timestr + ".json", "w", encoding="utf-8") as fout:
        json.dump(oaq_dim_per_disc, fout, ensure_ascii=False, indent=4)

import json
import os
import sys
import time

import pandas as pd
from convokit import Corpus, PolitenessStrategies, Speaker, TextParser, Utterance

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


def calculate_politeness(input_directory, moderator_flag=True):

    if not os.path.exists(input_directory):
        print(input_directory)
        print("input directory does not exist. Exiting")
        sys.exit(1)

    discussions = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if os.path.isfile(os.path.join(input_directory, f))
    ]
    print("Building corpus of ", len(discussions), "discussions")
    timestr = time.strftime("%Y%m%d-%H%M%S")

    politeness_per_disc = {}
    for disc in discussions:
        utterances, conv_topic = processDiscussion(disc, moderator_flag)
        disc_id = utterances[0].id.split("_")[2]
        corpus = Corpus(utterances=utterances)
        print("Corpus created successfully.")
        parser = TextParser()
        corpus = parser.transform(corpus)
        politeness_transformer = PolitenessStrategies()
        corpus = politeness_transformer.transform(corpus, markers=True)
        data = politeness_transformer.summarize(corpus, plot=False)
        politeness_per_disc[disc_id] = dict(data)

    average_politeness = []
    for disc_id, plt_features in politeness_per_disc.items():
        plt_df = pd.DataFrame(plt_features, index=[0])
        cols = plt_df.columns
        plt_df[cols] = plt_df[cols].apply(pd.to_numeric, errors="coerce")
        mean = plt_df.mean()
        average_politeness.append({disc_id: mean.to_dict()})

    flattened_data = []
    for item in average_politeness:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()

    average_politeness.append({"aggregate_mean": mean_values.to_dict()})

    with open("average_politeness_" + timestr + ".json", "w", encoding="utf-8") as fout:
        json.dump(average_politeness, fout, ensure_ascii=False, indent=4)
    """    
    with open('politeness_per_disc_'+timestr+'.json', 'w',encoding="utf-8") as fout:
        json.dump(politeness_per_disc, fout, ensure_ascii=False, indent=4)
    """

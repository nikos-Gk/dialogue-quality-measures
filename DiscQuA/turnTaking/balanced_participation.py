import json
import math
import os
import sys
import time

import pandas as pd
from convokit import Speaker, Utterance

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
    return utterances, conv_topic, speakers


def compute_balance(contributions):
    #    Compute the balance (S) of conversational contributions using entropy.

    #    Parameters:
    #    contributions (list): A list of contributions by each participant.

    #    Returns:
    #    float: The balance (S) of the conversation.

    # Calculate the total contributions
    total_contributions = sum(contributions)

    # Calculate the proportion of contributions for each participant
    proportions = [c / total_contributions for c in contributions]

    # Compute the entropy (balance)
    balance = -sum(p * math.log(p, len(proportions)) for p in proportions if p > 0)

    return balance


def calculate_balanced_participation(input_directory, moderator_flag=True):
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

    disc_speakers_messages_dict = {}
    disc_number_of_messages_dict = {}
    disc_sum_number_of_words_dict = {}
    for disc in discussions:
        print(disc)
        utterances, conv_topic, speakers = processDiscussion(disc, moderator_flag)
        disc_id = utterances[0].id.split("_")[2]
        print("Balanced Participation-Proccessing discussion: ", disc_id)
        disc_speakers_messages_dict[disc_id] = {
            speaker: [] for speaker in speakers.keys()
        }
        disc_number_of_messages_dict[disc_id] = []
        disc_sum_number_of_words_dict[disc_id] = []
        for utt in utterances:
            speaker = utt.get_speaker().id
            disc_speakers_messages_dict[disc_id][speaker].append(utt.text)
        for speaker in disc_speakers_messages_dict[disc_id].keys():
            speaker_words_per_message = []
            speaker_number_of_messages = len(
                disc_speakers_messages_dict[disc_id][speaker]
            )
            disc_number_of_messages_dict[disc_id].append(speaker_number_of_messages)
            for message in disc_speakers_messages_dict[disc_id][speaker]:
                speaker_words_per_message.append(len(message.split()))
            disc_speaker_number_of_words = sum(speaker_words_per_message)
            disc_sum_number_of_words_dict[disc_id].append(disc_speaker_number_of_words)

    with open(
        "number_of_messages_per_discussion_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(disc_number_of_messages_dict, fout, ensure_ascii=False, indent=4)

    with open(
        "sum_of_words_per_discussion_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(disc_sum_number_of_words_dict, fout, ensure_ascii=False, indent=4)

    entropy_per_discussion = {}
    for disc_id in disc_speakers_messages_dict.keys():
        entropy_per_discussion[disc_id] = {}
        entropy_per_discussion[disc_id]["entropy_number_of_messages"] = compute_balance(
            disc_number_of_messages_dict[disc_id]
        )
        entropy_per_discussion[disc_id]["entropy_number_of_words"] = compute_balance(
            disc_sum_number_of_words_dict[disc_id]
        )

    with open(
        "entropy_per_discussion_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(entropy_per_discussion, fout, ensure_ascii=False, indent=4)

    flattened_data_number_of_messages = []
    flattened_data_number_of_words = []
    for entropy_features_dict in entropy_per_discussion.values():
        for feature_entropy, value in entropy_features_dict.items():
            if feature_entropy == "entropy_number_of_messages":
                flattened_data_number_of_messages.append(value)
            elif feature_entropy == "entropy_number_of_words":
                flattened_data_number_of_words.append(value)
            else:
                print("error in flattened data")
                break
    df = pd.DataFrame(
        {
            "entropy_number_of_messages": flattened_data_number_of_messages,
            "entropy_number_of_words": flattened_data_number_of_words,
        }
    )
    mean_values = df.mean()

    entropy_per_discussion["aggregate_mean"] = mean_values.to_dict()

    with open(
        "entropy_per_discussion_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(entropy_per_discussion, fout, ensure_ascii=False, indent=4)

import io
import json
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
from convokit import Coordination, Corpus, Speaker, Utterance
from dateutil.relativedelta import relativedelta
from nltk.tokenize import sent_tokenize

#################################################################################
MODERATOR = "moderator"
MESSAGE_THREASHOLD_IN_CHARS = 25
HARDCODED_MODEL = "hardcoded"


def assign_sentence_to_speakers(sentence_text, speakers, previous_speaker):
    assigned_speakers = []
    for speaker in speakers.keys():
        if speaker in sentence_text:
            assigned_speakers.append(speaker)
    if len(assigned_speakers) == 0:
        assigned_speakers = previous_speaker
    return assigned_speakers


def processDiscussion(discussion, moderator_flag):
    with open(discussion, "r", encoding="utf-8") as file:
        data = json.load(file)

    conversation_id = data["id"]
    print(f"conversation_id:{conversation_id}")
    speakers = {user: Speaker(id=user) for user in data["users"]}

    if not moderator_flag:
        data["logs"] = [
            (log[0], log[1], log[2], log[3])
            for log in data["logs"]
            if log[0] != MODERATOR
        ]

    data["logs"] = [
        (log[0], log[1], log[2], log[3])
        for log in data["logs"]
        if ((isinstance(log[1], str)) and (len(log[1]) > MESSAGE_THREASHOLD_IN_CHARS))
    ]

    utterances = []
    utterances_history = {}
    reply_to_dict = {}
    data_cord = {"logs": []}
    data_cord_counter = -1
    for i, log in enumerate(data["logs"]):
        current_speaker, original_text, model, mid = log
        if i == 0:
            data_cord["logs"].append((current_speaker, original_text, model, mid, "-"))
            data_cord_counter += 1
            continue
        sentences = sent_tokenize(original_text)
        sentence_dict = {}
        for sentence_index, sentence_text in enumerate(sentences):
            if sentence_index == 0:
                assigned_speakers = assign_sentence_to_speakers(
                    sentence_text, speakers, []
                )
            else:
                assigned_speakers = assign_sentence_to_speakers(
                    sentence_text, speakers, assigned_speakers
                )
            sentence_dict[sentence_index] = assigned_speakers

        users_text = {}
        for sentence_index, sentence_text in enumerate(sentences):
            assinged_users = sentence_dict[sentence_index]

            if len(assinged_users) == 0:
                previous_speaker, text, model, mid, replyto_user = data_cord["logs"][
                    data_cord_counter
                ]
                assinged_users.append(previous_speaker)

            for user in assinged_users:
                if user in users_text.keys():
                    users_text[user] = users_text[user] + "\n" + sentence_text
                else:
                    users_text[user] = sentence_text

        for user, text in users_text.items():
            data_cord["logs"].append((current_speaker, text, model, mid, user))
            data_cord_counter += 1

    data_cord["logs"] = [
        (log[0], log[1], log[2], log[3], log[4])
        for log in data_cord["logs"]
        if log[0] != log[4]  # disregard users who reply to themselves
    ]

    strTime = time.strftime("%Y%m%d-%H%M%S")
    tm = datetime.strptime(strTime, "%Y%m%d-%H%M%S")
    for i, log in enumerate(data_cord["logs"]):
        speaker, text, model, mid, replyto_user = log
        utterances_history[speaker] = i
        tm = tm + relativedelta(seconds=1)
        if i == 0:
            reply_to_id = None
        else:
            utt_id = utterances_history[replyto_user]
            reply_to_id = "utt_" + str(utt_id) + "_" + str(conversation_id)

        #        if speaker == replyto_user:
        #            reply_to_id=None

        g = Utterance(
            id=f"utt_{i}_{conversation_id}",
            speaker=speakers[speaker],
            conversation_id=str(conversation_id),
            reply_to=reply_to_id,
            text=text,
            # meta={'timestamp': data['timestamp']})
            meta={"timestamp": tm},
        )
        g.timestamp = tm
        utterances.append(g)
        reply_to_dict[i] = reply_to_id
    return (
        utterances,
        speakers,
        data,
        utterances_history,
        data_cord,
        conversation_id,
        reply_to_dict,
    )


def save_stdout_to_image(func, conversation_id, *args, **kwargs):
    # Capture the standard output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    output = buffer.getvalue()
    fig, ax = plt.subplots()
    ax.text(
        0,
        1,
        output,
        fontsize=12,
        ha="left",
        va="top",
        wrap=True,
        transform=ax.transAxes,
    )
    ax.axis("off")
    folder_path = "output_images/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path + conversation_id + ".png", bbox_inches="tight")


def make_visualization(input_directory, moderator_flag=True):

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

    for disc in discussions:
        (
            utterances,
            speakers,
            data,
            utterances_history,
            data_cord,
            conversation_id,
            reply_to_dict,
        ) = processDiscussion(disc, moderator_flag)
        corpus = Corpus(utterances=utterances)
        print("Corpus created successfully.")
        corpus.print_summary_stats()
        conv0 = corpus.get_conversation(conversation_id)
        save_stdout_to_image(conv0.print_conversation_structure, conversation_id)

import io
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
from convokit import Corpus, Speaker, Utterance
from dateutil.relativedelta import relativedelta

from DiscQuA.utils import dprint


def save_stdout_to_image(func, conversation_id, *args, **kwargs):
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


def make_visualization(message_list, speakers_list, msgsid_list, replyto_list, disc_id):
    """Generates a text-based visualization of a discussion's structure and saves it as an image file.


    Args:
        message_list (list[str]): The list of utterances in the discussion.
        speakers_list (list[str]): The corresponding list of speakers for each utterance.
        msgsid_list (list[str]): List of messages ids corresponding to each utterance.
        replyto_list (list[str]): List indicating the message ID each utterance is replying to.
        disc_id (str): Unique identifier for the discussion.

    Returns:
        _type_: _description_
    """
    dprint("info", f"Building corpus of: {len(message_list)} utterances ")

    speakers_unq = set(speakers_list)
    speakers = {speaker: Speaker(id=speaker) for speaker in speakers_unq}
    utterances = []
    counter = 0
    timestr = time.strftime("%Y%m%d-%H%M%S")
    tm = datetime.strptime(timestr, "%Y%m%d-%H%M%S")
    for utt, speaker, msg_id, rplt in zip(
        message_list, speakers_list, msgsid_list, replyto_list
    ):
        tm = tm + relativedelta(seconds=1)
        if counter == 0:
            replyto = None
        else:
            replyto = str(rplt)
        u = Utterance(
            id=f"{msg_id}",
            speaker=speakers[speaker],
            conversation_id=str(disc_id),
            reply_to=replyto,
            text=utt,
            meta={"timestamp": tm},
        )
        u.timestamp = tm
        utterances.append(u)
        counter += 1

    corpus = Corpus(utterances=utterances)
    dprint("info", "Corpus created successfully.")
    # corpus.print_summary_stats()
    conv0 = corpus.get_conversation(disc_id)
    save_stdout_to_image(
        conv0.print_conversation_structure,
        disc_id,
        # lambda utt: utt.id + f"-{utt.speaker.id}",
        lambda utt: f"{utt.speaker.id}",
    )

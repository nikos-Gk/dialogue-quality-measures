import time
from datetime import datetime

from convokit import Corpus, HyperConvo, Speaker, Utterance
from dateutil.relativedelta import relativedelta

from DiscQuA.utils import dprint, save_dict_2_json


def calculate_structure_features(
    message_list, speakers_list, msgsid_list, replyto_list, disc_id, discussion_level
):
    """Extracts structural features from a discussion.Features can be computed over the full discussion or incrementally per utterance.

    Args:
        message_list (list[str]): The list of utterances in the discussion.
        speakers_list (list[str]): The corresponding list of speakers for each utterance.
        msgsid_list (list[str]): List of messages ids corresponding to each utterance.
        replyto_list (list[str]): List indicating the message ID each utterance is replying to.
        disc_id (str): Unique identifier for the discussion.
        discussion_level (bool): A boolean flag; if True, calculates structure features over the full discussion; otherwise if False, incrementally per utterance.


    Returns:
        tuple: Two dictionaries:
            - motifs_dict (dict): Contains reciprocity features extracted from the conversation structure.
            - features_dict (dict): Contains full structural feature vectors per discussion or per utterance span.
    """
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
    if discussion_level:
        corpus = Corpus(utterances=utterances)
        dprint("info", "Corpus created successfully.")
        # corpus.print_summary_stats()
        #############################################################################################################
        # prefix_len – Use the first [prefix_len] utterances of each conversation to construct the hypergraph
        # min_convo_len – Only consider conversations of at least this length
        hc = HyperConvo(prefix_len=40, min_convo_len=2)
        hc.transform(corpus)
        #############################################################################################################
        dt_features = corpus.get_vector_matrix("hyperconvo").to_dataframe()
        dt_features = dt_features.fillna(-1)
        feat_names = list(dt_features.columns)
        motif_count_feats = [
            x for x in feat_names if ("count" in x) and ("mid" not in x)
        ]
        #############################################################################################################
        motifs_dt = dt_features[motif_count_feats]
        motifs_dt_tp = motifs_dt.transpose()
        motifs_dict = motifs_dt_tp.to_dict()
        #############################################################################################################
        dt_features_tp = dt_features.transpose()
        features_dict = dt_features_tp.to_dict()
        #############################################################################################################
        save_dict_2_json(motifs_dict, "reciprocity_per_discussion_", disc_id, timestr)
        save_dict_2_json(
            features_dict, "structure_fet_per_discussion_", disc_id, timestr
        )
        #############################################################################################################
        return motifs_dict, features_dict
    else:
        output_dict_rec = {}
        output_dict_struct = {}
        for utter_index, utter in enumerate(utterances):
            if utter_index == 0:
                continue
            key_iter = "utt_" + str(0) + "-" + str(utter_index)
            corpus = Corpus(utterances=utterances[0 : utter_index + 1])
            # print(f"Corpus for utterances 0 - {utter_index} created successfully.")
            # corpus.print_summary_stats()
            hc = HyperConvo(prefix_len=40, min_convo_len=2)
            hc.transform(corpus)
            #############################################################################################################
            dt_features = corpus.get_vector_matrix("hyperconvo").to_dataframe()
            dt_features = dt_features.fillna(-1)
            feat_names = list(dt_features.columns)
            motif_count_feats = [
                x for x in feat_names if ("count" in x) and ("mid" not in x)
            ]
            motifs_dt = dt_features[motif_count_feats]
            motifs_dt_tp = motifs_dt.transpose()
            motifs_dict = motifs_dt_tp.to_dict()
            #
            dt_features_tp = dt_features.transpose()
            features_dict = dt_features_tp.to_dict()
            #
            if disc_id in output_dict_rec:
                output_dict_rec[disc_id].append(
                    [{key_iter: list(motifs_dict.values())}]
                )
            else:
                output_dict_rec[disc_id] = [{key_iter: list(motifs_dict.values())}]
            if disc_id in output_dict_struct:
                output_dict_struct[disc_id].append(
                    [{key_iter: list(features_dict.values())}]
                )
            else:
                output_dict_struct[disc_id] = [{key_iter: list(features_dict.values())}]
        #############################################################################################################
        save_dict_2_json(output_dict_rec, "reciprocity_per_ut_", disc_id, timestr)
        save_dict_2_json(output_dict_struct, "structure_fet_per_ut_", disc_id, timestr)
        #############################################################################################################
        return output_dict_rec, output_dict_struct

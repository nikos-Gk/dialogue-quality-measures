import math
import time

from DiscQuA.utils import getUtterances, save_dict_2_json


def compute_balance(contributions):
    total_contributions = sum(contributions)
    proportions = [c / total_contributions for c in contributions]
    balance = -sum(p * math.log(p, len(proportions)) for p in proportions if p > 0)
    return balance


def calculate_balanced_participation(
    message_list, speakers_list, disc_id, discussion_level
):
    utterances, speakers = getUtterances(message_list, speakers_list, disc_id)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if discussion_level:
        disc_speakers_messages_dict = {}
        disc_number_of_messages_dict = {}
        disc_sum_number_of_words_dict = {}
        print("Balanced Participation-Proccessing discussion: ", disc_id)
        disc_speakers_messages_dict[disc_id] = {
            speaker: [] for speaker in speakers.keys()
        }
        disc_number_of_messages_dict[disc_id] = {
            speaker: [] for speaker in speakers.keys()
        }
        disc_sum_number_of_words_dict[disc_id] = {
            speaker: [] for speaker in speakers.keys()
        }
        # dict with the general info:disc_speakers_messages_dict
        for utt in utterances:
            speaker = utt.get_speaker().id
            disc_speakers_messages_dict[disc_id][speaker].append(utt.text)
        # number of messages per speaker:disc_number_of_messages_dict
        for speaker in disc_speakers_messages_dict[disc_id].keys():
            #
            speaker_number_of_messages = len(
                disc_speakers_messages_dict[disc_id][speaker]
            )
            disc_number_of_messages_dict[disc_id][speaker] = speaker_number_of_messages
            #
            speaker_words_per_message = []
            #
            for message in disc_speakers_messages_dict[disc_id][speaker]:
                speaker_words_per_message.append(len(message.split()))
            disc_speaker_number_of_words = sum(speaker_words_per_message)
            disc_sum_number_of_words_dict[disc_id][
                speaker
            ] = disc_speaker_number_of_words
        save_dict_2_json(
            disc_number_of_messages_dict,
            "number_of_messages_per_discussion",
            disc_id,
            timestr,
        )
        save_dict_2_json(
            disc_sum_number_of_words_dict,
            "sum_of_words_per_discussion",
            disc_id,
            timestr,
        )
        entropy_per_discussion = {}
        for disc_id in disc_speakers_messages_dict.keys():
            entropy_per_discussion[disc_id] = {}
            entropy_per_discussion[disc_id]["entropy_number_of_messages"] = (
                compute_balance(disc_number_of_messages_dict[disc_id].values())
            )
            entropy_per_discussion[disc_id]["entropy_number_of_words"] = (
                compute_balance(disc_sum_number_of_words_dict[disc_id].values())
            )
            save_dict_2_json(
                entropy_per_discussion, "entropy_per_discussion", disc_id, timestr
            )
        return entropy_per_discussion
    else:
        output_dict_turn = {}
        output_dict_all = {}
        entropy_dict_all = {}
        for utter_index, utter in enumerate(utterances):
            if utter_index == 0:
                continue
            utterances_iter = utterances[0 : utter_index + 1]
            # print(f"Corpus for utterances 0 - {utter_index} created successfully.")
            #
            disc_speakers_messages_dict_temp = {}
            disc_number_of_messages_dict_temp = {}
            disc_sum_number_of_words_dict_temp = {}

            speakers_iter = [utt.get_speaker().id for utt in utterances_iter]

            #
            disc_speakers_messages_dict_temp[disc_id] = {
                speaker: [] for speaker in speakers_iter
            }
            disc_number_of_messages_dict_temp[disc_id] = {
                speaker: [] for speaker in speakers_iter
            }
            disc_sum_number_of_words_dict_temp[disc_id] = {
                speaker: [] for speaker in speakers_iter
            }

            #
            for utt in utterances_iter:
                speaker = utt.get_speaker().id
                disc_speakers_messages_dict_temp[disc_id][speaker].append(utt.text)
            #
            for speaker in disc_speakers_messages_dict_temp[disc_id].keys():
                #
                speaker_number_of_messages = len(
                    disc_speakers_messages_dict_temp[disc_id][speaker]
                )
                disc_number_of_messages_dict_temp[disc_id][
                    speaker
                ] = speaker_number_of_messages
                #
                speaker_words_per_message = []
                #
                for message in disc_speakers_messages_dict_temp[disc_id][speaker]:
                    #                    print('before splitting')
                    speaker_words_per_message.append(len(message.split()))
                #                    print('after splitting')
                disc_speaker_number_of_words = sum(speaker_words_per_message)
                disc_sum_number_of_words_dict_temp[disc_id][
                    speaker
                ] = disc_speaker_number_of_words
            key_iter = "utt_" + str(0) + "-" + str(utter_index)
            #
            output_dict_turn[key_iter] = {
                "number_of_messages": list(disc_number_of_messages_dict_temp.values()),
                "sum_of_words": list(disc_sum_number_of_words_dict_temp.values()),
            }
            if disc_id in output_dict_all:
                output_dict_all[disc_id].append([output_dict_turn])
            else:
                output_dict_all[disc_id] = [output_dict_turn]

            #
            entropy_per_turn = {}
            for disc_id in disc_speakers_messages_dict_temp.keys():
                entropy_per_turn[key_iter] = {}
                entropy_per_turn[key_iter]["entropy_number_of_messages"] = (
                    compute_balance(disc_number_of_messages_dict_temp[disc_id].values())
                )

                entropy_per_turn[key_iter]["entropy_number_of_words"] = compute_balance(
                    disc_sum_number_of_words_dict_temp[disc_id].values()
                )

            if disc_id in entropy_dict_all:
                entropy_dict_all[disc_id].append([entropy_per_turn])
            else:
                entropy_dict_all[disc_id] = [entropy_per_turn]
        save_dict_2_json(
            output_dict_all, "number_of_msgs_sumofwords_per_utt", disc_id, timestr
        )
        save_dict_2_json(entropy_dict_all, "entropy_per_utt", disc_id, timestr)
        return entropy_dict_all

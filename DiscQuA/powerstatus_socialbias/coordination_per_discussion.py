import time
from datetime import datetime

from convokit import Coordination, Corpus, Speaker, Utterance
from dateutil.relativedelta import relativedelta

from DiscQuA.utils import getUtterances, save_dict_2_json


def calculate_coordination_per_discussion(
    message_list, speakers_list, msgsid_list, replyto_list, disc_id, discussion_level
):
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
        output_dict_disc = {}
        corpus = Corpus(utterances=utterances)
        print("Corpus created successfully.")
        # corpus.print_summary_stats()
        # conv=corpus.get_conversation(disc_id)
        # conv.print_conversation_structure()
        coord = Coordination()
        coord.fit(corpus)
        coord.transform(corpus)
        #
        all_speakers = lambda speaker: True
        user_lamda_list = []

        for person in speakers.keys():
            user_lamda_list.append(
                {person: (lambda speaker, person=person: speaker.id == person)}
            )

        print("coordination of all speakers to user")
        coord_allspeakers_2_user = {}
        for item in user_lamda_list:
            user = list(item.keys())[0]
            lamda_function = list(item.values())[0]
            print("> coordination of all speakers to user ", user)
            allspeakers_2_user = coord.summarize(
                corpus,
                all_speakers,
                lamda_function,
                focus="targets",
                target_thresh=1,
                summary_report=True,
            )
            coord_allspeakers_2_user[user] = allspeakers_2_user

        print("coordination of user to all speakers")
        coord_user_2_allspeaker = {}
        for item in user_lamda_list:
            user = list(item.keys())[0]
            lamda_function = list(item.values())[0]
            print(f"> coordination of user {user} to all speakers ")
            user_2_allspeakers = coord.summarize(
                corpus,
                lamda_function,
                all_speakers,
                focus="speakers",
                speaker_thresh=1,
                target_thresh=1,
                summary_report=True,
            )
            coord_user_2_allspeaker[user] = user_2_allspeakers
        output_dict_disc[disc_id] = {
            "coord_allspeakers_2_user": coord_allspeakers_2_user,
            "coord_user_2_allspeaker": coord_user_2_allspeaker,
        }
        save_dict_2_json(output_dict_disc, "coord_per_discussion", disc_id, timestr)
        return output_dict_disc

    else:
        output_dict_all = {}
        output_dict_turn = {}
        for utter_index, utter in enumerate(utterances):
            if utter_index == 0:
                continue
            corpus = Corpus(utterances=utterances[0 : utter_index + 1])
            print(f"Corpus for utterances 0 - {utter_index} created successfully.")
            corpus.print_summary_stats()
            coord = Coordination()
            coord.fit(corpus)
            coord.transform(corpus)
            all_speakers = lambda speaker: True
            user_lamda_list = []
            for person in speakers.keys():
                user_lamda_list.append(
                    {person: (lambda speaker, person=person: speaker.id == person)}
                )
            print("coordination of all speakers to user")
            coord_allspeakers_2_user = {}
            for item in user_lamda_list:
                user = list(item.keys())[0]
                lamda_function = list(item.values())[0]
                print("> coordination of all speakers to user ", user)
                allspeakers_2_user = coord.summarize(
                    corpus,
                    all_speakers,
                    lamda_function,
                    focus="targets",
                    target_thresh=1,
                    summary_report=True,
                )
                coord_allspeakers_2_user[user] = allspeakers_2_user
            print("coordination of user to all speakers")
            coord_user_2_allspeaker = {}
            for item in user_lamda_list:
                user = list(item.keys())[0]
                lamda_function = list(item.values())[0]
                print(f"> coordination of user {user} to all speakers ")
                user_2_allspeakers = coord.summarize(
                    corpus,
                    lamda_function,
                    all_speakers,
                    focus="speakers",
                    speaker_thresh=1,
                    target_thresh=1,
                    summary_report=True,
                )
                coord_user_2_allspeaker[user] = user_2_allspeakers
            key_iter = "utt_" + str(0) + "-" + str(utter_index)
            output_dict_turn[key_iter] = {
                "coord_allspeakers_2_user": coord_allspeakers_2_user,
                "coord_user_2_allspeaker": coord_user_2_allspeaker,
            }
        if disc_id in output_dict_all:
            output_dict_all[disc_id].append([output_dict_turn])
        else:
            output_dict_all[disc_id] = [output_dict_turn]
        save_dict_2_json(output_dict_all, "coord_per_utt", disc_id, timestr)
        return output_dict_all

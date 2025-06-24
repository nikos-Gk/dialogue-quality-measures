import time

from convokit import Corpus, PolitenessStrategies, TextParser

from DiscQuA.utils import getUtterances, save_dict_2_json


def calculate_politeness(message_list, speakers_list, disc_id, discussion_level):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    utterances, speakers = getUtterances(message_list, speakers_list, disc_id)
    if discussion_level:
        corpus = Corpus(utterances=utterances)
        print("Corpus created successfully.")
        # corpus.print_summary_stats()
        politeness_per_disc = {}
        parser = TextParser()
        corpus = parser.transform(corpus)
        politeness_transformer = PolitenessStrategies()
        corpus = politeness_transformer.transform(corpus, markers=True)
        data = politeness_transformer.summarize(corpus, plot=False)
        politeness_per_disc[disc_id] = dict(data)
        save_dict_2_json(politeness_per_disc, "politeness_per_disc", disc_id, timestr)
        return politeness_per_disc
    else:
        politenes_per_utt = {}
        for utterance_index, utterance_iter in enumerate(utterances):
            corpus = Corpus(
                utterances=utterances[utterance_index : utterance_index + 1]
            )
            # print("Corpus for utterances created successfully.")
            # corpus.print_summary_stats()
            parser = TextParser()
            corpus = parser.transform(corpus)
            politeness_transformer = PolitenessStrategies()
            corpus = politeness_transformer.transform(corpus, markers=True)
            data = politeness_transformer.summarize(corpus, plot=False)
            if disc_id in politenes_per_utt:
                politenes_per_utt[disc_id].append(
                    {"utt_" + str(utterance_index): dict(data)}
                )
            else:
                politenes_per_utt[disc_id] = [
                    {"utt_" + str(utterance_index): dict(data)}
                ]
        save_dict_2_json(politenes_per_utt, "politeness_per_utt", disc_id, timestr)
        return politenes_per_utt

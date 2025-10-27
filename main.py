# import unsloth
import json
import logging

# import os
from pprint import pprint

from discqua import (
    arg_dimensions,
    coherence_disc,
    coherence_ecoh,
    coherence_response,
    collaboration,
    controversy,
    coordination,
    dialogue_acts,
    dispute_tactics,
    diversity_disc,
    diversity_response,
    empathy,
    engagement_disc,
    engagement_response,
    informativeness_disc,
    informativeness_response,
    make_visualization,
    ngramdiversity,
    overall_arg_quality,
    participation,
    persuasion_strategy,
    persuasiveness_disc,
    politeness,
    politeness_ngrams,
    readability,
    reciprocity,
    sentiment,
    social_bias,
    toxicity,
    utils,
)

# model_path = r"D:\virtual\models\llama-2-13b-chat.Q5_K_M.gguf"
model_path = "unsloth/Meta-Llama-3.1-8B-Instruct"
# "unsloth/Meta-Llama-3.1-8B-Instruct" #"unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

test_path = "./example_discussion/fe858614-6571-43e7-badf-3cdcb38337b3.json"

with open(test_path, "r", encoding="utf-8") as file:
    data = json.load(file)

utils.set_saving_enabled(True)
saving_path = ""
utils.set_output_path(saving_path)

message_list = [log[1] for log in data["logs"]]
speakers_list = [log[0] for log in data["logs"]]
disc_id = "fe858614-6571-43e7-badf-3cdcb38337b3"
#
msgsid_list = [log[3] for log in data["logs"]]
replyto_list = []
replyto_list.append("-")
aux = [msg for msg in msgsid_list]
for i, v in enumerate(aux):
    replyto_list.append(v)
    if i == len(msgsid_list) - 1:
        break


def test_argument_qual_dim():
    arg_dim_scores = arg_dimensions(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        dimension="logic",
        # device="cuda:0"
    )
    # print(arg_dim_scores)


def test_overall_arg_quality():
    overall_argument_quality_score = overall_arg_quality(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        mode="rating",
        gpu=True,
        # device="cuda:0"
    )
    # print(overall_argument_quality_score)


def test_coherence_conversation():
    coherence_disc_score = coherence_disc(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        # device="cuda:0"
    )
    # print(coherence_disc_score)


def test_coherence_response():
    coherence_resp_score = coherence_response(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    # print(coherence_resp_score)


def test_coherence_ecoh():
    coherence_booleans = coherence_ecoh(
        message_list=message_list,
        speaker_list=speakers_list,
        disc_id=disc_id,
        device="cuda",
    )
    # print(coherence_booleans)


def test_collaboration():
    collaboration_scores = collaboration(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        discussion_level=False,
    )
    # print(collaboration_scores)


def test_controversy():
    unorm_scores, norm_scores = controversy(
        message_list=message_list,
        disc_id=disc_id,
        msgsid_list=msgsid_list,
        discussion_level=True,
        # device="cuda",
    )
    # pprint(unorm_scores)
    # pprint(norm_scores)


def test_dialogue_acts():
    dialogue_acts_scores = dialogue_acts(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    # print(dialogue_acts_scores)


def test_dispute_tactics():
    disp_tact = dispute_tactics(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    # print(disp_tact)


def test_diversity_conversation():
    diversity_disc_score = diversity_disc(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        # device="cuda:0"
    )
    # print(diversity_disc_score)


def test_diversity_response():
    diversity_resp_score = diversity_response(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    # print(diversity_resp_score)


def test_ngram_diversity():
    lexical_diversity = ngramdiversity(
        message_list=message_list, msgsid_list=msgsid_list, disc_id=disc_id
    )
    # print(lexical_diversity)


def test_expressed_empathy():
    expr_empathy_labels = empathy(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    # print(expr_empathy_labels)


def test_engagement_conversation():
    engagement_disc_score = engagement_disc(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        # device="cuda:0"
    )
    # print(engagement_disc_score)


def test_engagement_response():
    engagement_resp_score = engagement_response(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    # print(engagement_resp_score)


def test_reciprocity():
    reciprocity_features = reciprocity(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        replyto_list=replyto_list,
        disc_id=disc_id,
        discussion_level=False,
    )
    # pprint(reciprocity_features)


def test_informativeness_conversation():
    informativeness_disc_score = informativeness_disc(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        # device="cuda:0"
    )
    # print(informativeness_disc_score)


def test_informativeness_response():
    informativeness_resp_score = informativeness_response(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    # print(informativeness_resp_score)


def test_persuasiveness():
    persuasiveness_score = persuasiveness_disc(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        # device="cuda:0"
    )
    # print(persuasiveness_score)


def test_persuasion_strategy():
    pers_strategy = persuasion_strategy(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    # print(pers_strategy)


def test_politeness_analysis():
    politeness_score = politeness(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    # print(politeness_score)


def test_politeness():
    politeness_features_scores = politeness_ngrams(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        discussion_level=True,
    )
    # print(politeness_features_scores)


def test_social_bias():
    social_bias_labels = social_bias(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    # print(social_bias_labels)


def test_coordination():
    coordination_scores = coordination(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        replyto_list=replyto_list,
        disc_id=disc_id,
        discussion_level=True,
    )
    # print(coordination_scores)


def test_readability():
    readability_scores = readability(
        message_list=message_list, msgsid_list=msgsid_list, disc_id=disc_id
    )
    # print(readability_scores)


def test_sentiment_analysis():
    sentiment_scores = sentiment(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    # print(sentiment_scores)


def test_toxicity():
    toxicity_scores = toxicity(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        conver_topic=message_list[0],
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    # print(toxicity_scores)


def test_balanced_participation():
    entropy_scores = participation(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        disc_id=disc_id,
        discussion_level=False,
    )
    # pprint(entropy_scores)


def test_make_turn_taking_visualization():
    make_visualization(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        replyto_list=replyto_list,
        disc_id=disc_id,
    )


def non_llm_measures():
    print("Non LLM measures")
    # test_reciprocity()
    # test_balanced_participation()
    # test_make_turn_taking_visualization()
    # test_ngram_diversity()
    # test_coordination()
    # test_politeness()
    # test_readability()
    # test_collaboration()


def llm_measures():
    print("LLM measures")
    # test_argument_qual_dim()
    # test_overall_arg_quality()
    # test_coherence_conversation()
    # test_coherence_response()
    # test_coherence_ecoh()
    # test_diversity_conversation()
    # test_diversity_response()
    # test_engagement_conversation()
    # test_engagement_response()
    # test_informativeness_conversation()
    # test_informativeness_response()
    # test_persuasiveness()
    # test_persuasion_strategy()
    # test_politeness_analysis()
    # test_toxicity()
    # test_social_bias()
    # test_sentiment_analysis()
    # test_dispute_tactics()
    # test_dialogue_acts()
    # test_controversy()
    # test_expressed_empathy()


if __name__ == "__main__":
    print("Executing discussion quality aspects.")
    logging.basicConfig(level=logging.ERROR)
    # non_llm_measures()
    # llm_measures()

# import unsloth
import json
from pprint import pprint

from DiscQuA import (
    calculate_arg_dim,
    calculate_balanced_participation,
    calculate_coherence_conversation,
    calculate_coherence_ecoh,
    calculate_coherence_response,
    calculate_collaboration,
    calculate_controversy,
    calculate_coordination_per_disc_utt,
    calculate_dispute_tactics,
    calculate_diversity_conversation,
    calculate_diversity_response,
    calculate_engagement_conversation,
    calculate_engagement_response,
    calculate_informativeness_conversation,
    calculate_informativeness_response,
    calculate_language_features,
    calculate_overall_arg_quality,
    calculate_persuasiveness,
    calculate_politeness,
    calculate_readability,
    calculate_social_bias,
    calculate_speech_acts,
    calculate_structure_features,
    calculate_toxicity,
    dialogicity,
    make_visualization,
)

# model_path = r"D:\virtual\models\llama-2-13b-chat.Q5_K_M.gguf"
model_path = "unsloth/Meta-Llama-3.1-8B-Instruct"
# "unsloth/Meta-Llama-3.1-8B-Instruct" #"unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

test_path = "./example_discussion/fe858614-6571-43e7-badf-3cdcb38337b3.json"

with open(test_path, "r", encoding="utf-8") as file:
    data = json.load(file)

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


def test_structure_features():
    structure_features = calculate_structure_features(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        replyto_list=replyto_list,
        disc_id=disc_id,
        discussion_level=False,
    )
    pprint(structure_features)


def test_balanced_participation():
    entropy_scores = calculate_balanced_participation(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        discussion_level=False,
    )
    pprint(entropy_scores)


def test_make_turn_taking_visualization():
    make_visualization(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        replyto_list=replyto_list,
        disc_id=disc_id,
    )


def test_language_features():
    language_fetures_scores = calculate_language_features(
        message_list=message_list,
        disc_id=disc_id,
    )
    pprint(language_fetures_scores)


def test_controversy():
    unorm_scores, norm_scores = calculate_controversy(
        message_list=message_list, disc_id=disc_id, discussion_level=False
    )
    pprint(unorm_scores)
    pprint(norm_scores)


def test_coherence_conversation():
    coherence_disc_score = calculate_coherence_conversation(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        # device="cuda:0"
    )
    print(coherence_disc_score)


def test_coherence_response():
    coherence_resp_score = calculate_coherence_response(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    print(coherence_resp_score)


def test_coherence_ecoh():
    coherence_booleans = calculate_coherence_ecoh(
        message_list=message_list,
        speaker_list=speakers_list,
        disc_id=disc_id,
        device="cuda",
    )
    print(coherence_booleans)


def test_diversity_conversation():
    diversity_disc_score = calculate_diversity_conversation(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        # device="cuda:0"
    )
    print(diversity_disc_score)


def test_diversity_response():
    diversity_resp_score = calculate_diversity_response(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    print(diversity_resp_score)


def test_dialogicity():
    dialogicity_labels = dialogicity(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    print(dialogicity_labels)


def test_engagement_conversation():
    engagement_disc_score = calculate_engagement_conversation(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        # device="cuda:0"
    )
    print(engagement_disc_score)


def test_engagement_response():
    engagement_resp_score = calculate_engagement_response(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    print(engagement_resp_score)


def test_informativeness_conversation():
    informativeness_disc_score = calculate_informativeness_conversation(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        # device="cuda:0"
    )
    print(informativeness_disc_score)


def test_informativeness_response():
    informativeness_resp_score = calculate_informativeness_response(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    print(informativeness_resp_score)


def test_persuasiveness():
    persuasiveness_score = calculate_persuasiveness(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        # device="cuda:0"
    )
    print(persuasiveness_score)


def test_toxicity():
    toxicity_scores = calculate_toxicity(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    print(toxicity_scores)


def test_coordination_per_disc_utt():
    coordination_scores = calculate_coordination_per_disc_utt(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
        replyto_list=replyto_list,
        disc_id=disc_id,
        discussion_level=False,
    )
    print(coordination_scores)


def test_social_bias():
    social_bias_labels = calculate_social_bias(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    print(social_bias_labels)


def test_politeness():
    politeness_features_scores = calculate_politeness(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        discussion_level=False,
    )
    print(politeness_features_scores)


def test_readability():
    readability_scores = calculate_readability(
        message_list=message_list, disc_id=disc_id
    )
    print(readability_scores)


def test_collaboration():
    collaboration_scores = calculate_collaboration(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        discussion_level=False,
    )
    print(collaboration_scores)


def test_argument_qual_dim():
    arg_dim_scores = calculate_arg_dim(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        dimension="logic",
        # device="cuda:0"
    )
    print(arg_dim_scores)


def test_overall_arg_quality():
    overall_argument_quality_score = calculate_overall_arg_quality(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        mode="rating",
        gpu=True,
        # device="cuda:0"
    )
    print(overall_argument_quality_score)


def test_dispute_tactics():
    disp_tact = calculate_dispute_tactics(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    print(disp_tact)


def test_speech_acts():
    speech_acts = calculate_speech_acts(
        message_list,
        speakers_list,
        disc_id,
        openAIKEY="",
        model_type="transformers",
        model_path=model_path,
        gpu=True,
        ctx=1,
        # device="cuda:0"
    )
    print(speech_acts)


def non_llm_measures():
    print("Non LLM measures")
    # test_structure_features()
    # test_balanced_participation()
    # test_make_turn_taking_visualization()
    # test_language_features()
    # test_controversy()
    # test_coordination_per_disc_utt()
    # test_politeness()
    # test_readability()
    # test_collaboration()


def llm_measures():
    print("LLM measures")
    # test_coherence_conversation()  #
    # test_coherence_response()  #
    # test_coherence_ecoh()  #
    # test_diversity_conversation()  #
    # test_diversity_response()
    # test_dialogicity()  #
    # test_engagement_conversation()  #
    # test_engagement_response()
    # test_informativeness_conversation()  #
    # test_informativeness_response()
    # test_persuasiveness()  #
    # test_toxicity()  #
    # test_social_bias()  #
    # test_argument_qual_dim()  #
    # test_overall_arg_quality()  #
    # test_dispute_tactics()  #
    # test_speech_acts()


if __name__ == "__main__":
    print("Executing discussion quality aspects.")
    # non_llm_measures()
    # llm_measures()

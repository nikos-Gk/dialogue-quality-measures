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
    calculate_coordination_per_discussion,
    calculate_dispute_tactics,
    calculate_diversity,
    calculate_language_features,
    calculate_overall_arg_quality,
    calculate_politeness,
    calculate_social_bias,
    calculate_structure_features,
    dialogicity,
    engagement,
    informativeness,
    make_visualization,
    persuasiveness,
    toxicity,
)

# model_path = "meta-llama/Meta-Llama-3-8B-Instruct"  # "TheBloke/Llama-2-13B-chat-GGUF"  # "unsloth/Llama-3.2-1B-Instruct"  # r"D:\virtual\models\llama-2-13b-chat.Q5_K_M.gguf"
model_path = r"D:\virtual\models\llama-2-13b-chat.Q5_K_M.gguf"

test_path = r".\example_discussion\fe858614-6571-43e7-badf-3cdcb38337b3.json"
with open(test_path, "r", encoding="utf-8") as file:
    data = json.load(file)

message_list = [log[1] for log in data["logs"]]
speakers_list = [log[0] for log in data["logs"]]
disc_id = "fe858614-6571-43e7-badf-3cdcb38337b3"
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
        discussion_level=True,
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
        message_list=message_list, disc_id=disc_id
    )
    pprint(unorm_scores)
    pprint(norm_scores)


def test_coherence_conversation():
    coherence_disc_score = calculate_coherence_conversation(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="openai",
        model_path="",
        gpu=False,
    )
    print(coherence_disc_score)


def test_coherence_response():
    score = calculate_coherence_response(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="openai",
        model_path="",
        gpu=True,
        ctx=3,
    )
    print(score)


def test_coherence_ecoh():
    coherence_booleans = calculate_coherence_ecoh(
        message_list=message_list,
        speaker_list=speakers_list,
        disc_id=disc_id,
        device="cuda",
    )
    print(coherence_booleans)


def test_diversity():
    diversity_score = calculate_diversity(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="openai",
        model_path="",
        gpu=False,
    )
    print(diversity_score)


def test_dialogicity():
    dialogicity_labels = dialogicity(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="openai",
        model_path="",
        gpu=True,
        ctx=1,
    )
    print(dialogicity_labels)


def test_engagement():
    engagement_score = engagement(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="openai",
        model_path="",
        gpu=False,
    )
    print(engagement_score)


def test_informativeness():
    informativeness_score = informativeness(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="openai",
        model_path="",
        gpu=False,
    )
    print(informativeness_score)


def test_persuasiveness():
    persuasiveness_score = persuasiveness(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="openai",
        model_path="",
        gpu=False,
    )
    print(persuasiveness_score)


def test_toxicity():
    toxicity_scores = toxicity(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="openai",
        model_path="",
        gpu=False,
        ctx=1,
    )
    print(toxicity_scores)


def test_coordination_per_discussion():
    coordination_scores = calculate_coordination_per_discussion(
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
        model_type="openai",
        model_path="",
        gpu=False,
        ctx=1,
    )
    print(social_bias_labels)


def test_politeness():
    politeness_features_scores = calculate_politeness(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        discussion_level=True,
    )
    print(politeness_features_scores)


def test_collaboration():
    collaboration_scores = calculate_collaboration(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        discussion_level=True,
    )
    print(collaboration_scores)


def test_argument_qual_dim():
    arg_dim_scores = calculate_arg_dim(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="openai",
        model_path="",
        gpu=False,
        ctx=1,
    )
    print(arg_dim_scores)


def test_overall_arg_quality():
    overall_argument_quality_score = calculate_overall_arg_quality(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="openai",
        model_path="",
        mode="real",
        gpu=False,
    )
    print(overall_argument_quality_score)


def test_dispute_tactics():
    disp_tact = calculate_dispute_tactics(
        message_list=message_list,
        speakers_list=speakers_list,
        disc_id=disc_id,
        openAIKEY="",
        model_type="openai",
        model_path="",
        gpu=False,
        ctx=1,
    )
    print(disp_tact)


if __name__ == "__main__":
    print("Executing discussion quality aspects.")
    # test_structure_features()
    # test_balanced_participation()
    # test_make_turn_taking_visualization()
    # test_language_features()
    # test_controversy()
    # test_coherence_conversation()  # llama-cpp-python
    # test_coherence_response()  #
    # test_coherence_ecoh()  #
    # test_diversity()  #
    # test_dialogicity()  #
    # test_engagement()  #
    # test_informativeness()  #
    # test_persuasiveness()  #
    # test_toxicity()  #
    # test_coordination_per_discussion()
    # test_social_bias()  #
    # test_politeness()
    # test_collaboration()
    # test_argument_qual_dim()  #
    # test_overall_arg_quality()  #
    # test_dispute_tactics()  #

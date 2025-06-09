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
    convert_csv_to_json,
    dialogicity,
    engagement,
    informativeness,
    make_visualization,
    persuasiveness,
    toxicity,
)

disc_directory = "Directory with jsons. This will be produced after the execution of convert_csv_to_json()"
model_path = r"path\to\model\llama-2-13b-chat.Q5_K_M.gguf"


def test_csv_to_json():
    convert_csv_to_json(
        input_file="Path to csv file to extract conversations and create json files"
    )


def test_structure_features():
    calculate_structure_features(
        input_directory=disc_directory,
        moderator_flag=False,
    )


def test_balanced_participation():
    calculate_balanced_participation(
        input_directory=disc_directory,
        moderator_flag=False,
    )


def test_make_turn_taking_visualization():
    make_visualization(
        input_directory=disc_directory,
        moderator_flag=False,
    )


def test_language_features():
    calculate_language_features(
        input_directory=disc_directory,
        moderator_flag=False,
    )


def test_controversy():
    calculate_controversy(
        input_directory=disc_directory,
        moderator_flag=False,
    )


def test_coherence_conversation_openai():
    calculate_coherence_conversation(
        input_directory=disc_directory,
        openAIKEY="your-openai-key",
        model_type="openai",
        model_path="",
        gpu=True,
        moderator_flag=False,
    )


def test_coherence_conversation_llama():
    calculate_coherence_conversation(
        input_directory=disc_directory,
        openAIKEY="",
        model_type="llama",
        model_path=model_path,
        gpu=True,
        moderator_flag=False,
    )


def test_coherence_response():
    calculate_coherence_response(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )


def test_coherence_ecoh():
    calculate_coherence_ecoh(
        input_directory=disc_directory,
        moderator_flag=False,
        device="cpu",
    )


def test_diversity():
    calculate_diversity(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )


def test_dialogicity():
    dialogicity(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )


def test_engagement():
    engagement(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )


def test_informativeness():
    informativeness(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )


def test_persuasiveness():
    persuasiveness(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )


def test_toxicity():
    toxicity(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )


def test_coordination_per_discussion():
    calculate_coordination_per_discussion(
        input_directory=disc_directory,
        moderator_flag=False,
    )


def test_social_bias():
    calculate_social_bias(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )


def test_politeness():
    calculate_politeness(
        input_directory=disc_directory,
        moderator_flag=False,
    )


def test_collaboration():
    calculate_collaboration(
        input_directory=disc_directory,
        moderator_flag=False,
    )


def test_argument_qual_dim():
    calculate_arg_dim(
        input_directory=disc_directory, openAIKEY="", moderator_flag=False
    )


def test_overall_arg_quality():
    calculate_overall_arg_quality(
        input_directory=disc_directory,
        openAIKEY="",
        mode="real",
        moderator_flag=False,
    )


def test_dispute_tactics():
    calculate_dispute_tactics(
        input_directory=disc_directory, openAIKEY="", moderator_flag=False
    )


if __name__ == "__main__":
    print("Executing discussion quality aspects.")
    # test_csv_to_json()
    # test_structure_features()
    # test_balanced_participation()
    # test_make_turn_taking_visualization()
    # test_language_features()
    # test_controversy()
    # test_coherence_conversation_openai() #
    # test_coherence_conversation_llama() #
    # test_coherence_response() #
    # test_coherence_ecoh() #
    # test_diversity()  #
    # test_dialogicity() #
    # test_engagement() #
    # test_informativeness() #
    # test_persuasiveness()  #
    # test_toxicity() #
    # test_coordination_per_discussion()
    # test_social_bias() #
    # test_politeness()
    # test_collaboration()
    # test_argument_qual_dim() #
    # test_overall_arg_quality() #
    # test_dispute_tactics() #

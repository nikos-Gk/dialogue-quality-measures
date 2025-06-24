import time

from DiscQuA.utils import (
    getModel,
    getUtterances,
    save_dict_2_json,
    sleep,
    validateInputParams,
)

from .DisputeTacticsCl import DisputeTactics


def calculate_dispute_tactics(
    message_list,
    speakers_list,
    disc_id,
    openAIKEY,
    model_type="openai",
    model_path="",
    gpu=False,
    ctx=1,
):
    validateInputParams(model_type, openAIKEY, model_path)
    print("Building corpus of ", len(message_list), "utterances")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None

    if model_type == "llama":
        llm = getModel(model_path, gpu)

    dispute_tactics_llm_output_dict = {}

    try:
        utterances, speakers = getUtterances(
            message_list, speakers_list, disc_id, replyto_list=[]
        )
        conv_topic = message_list[0]
        print("Discpute tactics-Proccessing disc: ", disc_id, " with LLM")

        dispute = DisputeTactics(
            utterances, conv_topic, openAIKEY, model_type, llm, ctx
        )
        dispute_features = dispute.calculate_dispute_tectics()
        dispute_tactics_llm_output_dict[disc_id] = dispute_features

        sleep(model_type)
    except Exception as e:
        print("Error: ", e)
        print(disc_id)

    save_dict_2_json(
        dispute_tactics_llm_output_dict, "llm_dispute_tactics", disc_id, timestr
    )

    """        
    with open("llm_dispute_tactics_.json", encoding="utf-8") as f:
        d = json.load(f)
    dispute_tactics_llm_output_dict=d
    """

    dispute_tactics_per_disc = {}
    for disc_id, turnAnnotations in dispute_tactics_llm_output_dict.items():
        for label in turnAnnotations:
            if label == -1 or not label.startswith("- Level 0:"):
                print("LLM output for utterance is ill-formatted, skipping utterance\n")
                print(label)
                continue
            parts = label.split("\n")
            feature = {}
            try:
                for j in parts:
                    entries = j.split(":")
                    key = entries[0]
                    value = entries[1]
                    key = key.replace("-", "")
                    value = value.replace("[", "").replace("]", "")
                    feature[key] = value
                if disc_id in dispute_tactics_per_disc:
                    dispute_tactics_per_disc[disc_id].append(feature)
                else:
                    dispute_tactics_per_disc[disc_id] = [feature]
            except Exception as e:
                print(e)

    save_dict_2_json(
        dispute_tactics_per_disc,
        "dispute_tactics_features_per_utterance",
        disc_id,
        timestr,
    )
    return dispute_tactics_per_disc

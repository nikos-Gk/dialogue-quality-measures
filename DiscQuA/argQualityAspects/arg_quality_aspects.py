import time

from DiscQuA.utils import (
    getModel,
    getUtterances,
    save_dict_2_json,
    sleep,
    validateInputParams,
)

from .arg_qual_dimensions import AQualityDimensions


def calculate_arg_dim(
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

    argqualitydimensions_scores_llm_output_dict = {}

    try:
        utterances, speakers = getUtterances(
            message_list, speakers_list, disc_id, replyto_list=[]
        )
        conv_topic = message_list[0]
        print("Argument Quality Dimensions-Proccessing disc: ", disc_id, " with LLM")

        argqualdimensions = AQualityDimensions(
            utterances, conv_topic, openAIKEY, model_type, llm, ctx
        )

        argument_quality_dimensions_features = (
            argqualdimensions.argquality_dimensions_scores()
        )

        argqualitydimensions_scores_llm_output_dict[disc_id] = (
            argument_quality_dimensions_features
        )

        sleep(model_type)

    except Exception as e:
        print("Error: ", e)
        print(disc_id)

    save_dict_2_json(
        argqualitydimensions_scores_llm_output_dict, "llm_output_maq", disc_id, timestr
    )

    """        
    with open("llm_output_maq.json", encoding="utf-8") as f:
        maq = json.load(f)
    argqualitydimensions_scores_llm_output_dict=maq
    """
    arq_dim_per_disc = {}
    for disc_id, turnAnnotations in argqualitydimensions_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label == -1 or not label.startswith("-Level 1a:"):
                print("LLM output for utterance is ill-formatted, skipping utterance\n")
                print(label)
                continue
            parts = label.split("\n")
            if len(parts) != 15:
                print(
                    "LLM output with missing arg quality dimensions, skipping utterance\n"
                )
                print(label)
                continue
            feature = {}
            try:
                for j in parts:
                    entries = j.split(":")
                    key = entries[0]
                    value = entries[1]
                    key = key.replace("-", "")
                    rightParenthesisIndex = value.find("]")
                    leftParenthesisIndex = value.find("[")
                    value = value[leftParenthesisIndex:rightParenthesisIndex]
                    value = value.replace("[", "").replace("]", "")
                    feature[key] = value
                if disc_id in arq_dim_per_disc:
                    arq_dim_per_disc[disc_id].append(feature)
                else:
                    arq_dim_per_disc[disc_id] = [feature]
            except Exception as e:
                print(e)

    save_dict_2_json(arq_dim_per_disc, "maq_per_utterance", disc_id, timestr)
    return arq_dim_per_disc

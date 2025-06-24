import time

from DiscQuA.utils import (
    getModel,
    getUtterances,
    isValidResponse,
    save_dict_2_json,
    validateInputParams,
)

from .argument_quality_overall import OAQuality


def calculate_overall_arg_quality(
    message_list,
    speakers_list,
    disc_id,
    openAIKEY,
    model_type="openai",
    model_path="",
    mode="real",
    gpu=False,
):
    validateInputParams(model_type, openAIKEY, model_path)
    print("Building corpus of ", len(message_list), "utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None

    if model_type == "llama":
        llm = getModel(model_path, gpu)

    ovargquality_scores_llm_output_dict = {}
    #
    utterances, speakers = getUtterances(
        message_list, speakers_list, disc_id, replyto_list=[]
    )
    conv_topic = message_list[0]
    #
    print("Overall Argument Quality-Proccessing discussion: ", disc_id, " with LLM")
    ovargqual = OAQuality(utterances, conv_topic, openAIKEY, mode, model_type, llm)
    ovargument_quality_scores_features = ovargqual.calculate_ovargquality_scores()
    ovargquality_scores_llm_output_dict[disc_id] = ovargument_quality_scores_features
    #
    save_dict_2_json(
        ovargquality_scores_llm_output_dict, "llm_output_oaq", disc_id, timestr
    )

    """        
    with open("llm_output_oaq.json", encoding="utf-8") as f:
        oaq = json.load(f)
    ovargquality_scores_llm_output_dict=oaq
    """

    oaq_dim_per_disc = {}
    for disc_id, turnAnnotations in ovargquality_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label == -1:
                print(
                    "LLM output with missing overall argument quality , skipping discussion\n"
                )
                print(label)
                continue
            parts = label.split(
                "The average overall quality of the arguments presented in the above discussion is:"
            )
            value = isValidResponse(parts)
            if value == -1:
                print(
                    "LLM output with missing overall argument quality , skipping discussion\n"
                )
                print(label)
                continue

            oaq_dim_per_disc[disc_id] = value

    save_dict_2_json(oaq_dim_per_disc, "oaq_per_disc", disc_id, timestr)
    return oaq_dim_per_disc

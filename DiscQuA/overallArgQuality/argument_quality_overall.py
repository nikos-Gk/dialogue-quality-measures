# -*- coding: utf-8 -*-
from DiscQuA.utils import prompt_gpt4

#################################################################################

prompt_realnumber = """{conv_text}'\n\n\n

The texts above show a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating 
inflammatory and aggressive speech.
Evaluate the average overall quality of the arguments presented in the above discussion.
Although a definition of overall argument quality does not exist so far, try to judge about the overall quality based on all those argument quality dimensions that you think influence the overall quality of the given argumentation.
Assign a score for the average overall argument quality of the arguments presented in the entire discussion on a scale from 1 to 5, where 1 is of poor quality and 5 is of extremely good quality. 
Conclude your evaluation with the statement: 'The average overall quality of the arguments presented in the above discussion is: [X]', where X is the numeric score (real number) you've determined.
Please, ensure that your last statement is the score in brackets [].
"""

prompt_ratingscale = """{conv_text}'\n\n\n

The texts above show a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating 
inflammatory and aggressive speech.
Evaluate the average overall quality of the arguments presented in the above discussion.
Although a definition of overall argument quality does not exist so far, try to judge about the overall quality based on all those argument quality dimensions that you think influence the overall quality of the given argumentation.
Assign a label for the average overall argument quality of the arguments presented in the entire discussion on a scale from 1 to 3, where 1 is low quality, 2 is medium quality and 3 is high quality. 
Conclude your evaluation with the statement: 'The average overall quality of the arguments presented in the above discussion is: [X]', where X is the label you've determined.
Please, ensure that your last statement is the label in brackets [].

"""


class OAQuality:
    def __init__(self, utterances, conv_topic, openaiKey, mode, model_type, llm):
        self.utterances = utterances
        self.conv_topic = conv_topic
        self.openaiKey = openaiKey
        self.mode = mode
        self.model_type = model_type
        self.llm = llm

    def calculate_ovargquality_scores(self):
        mode_prompt = self.mode
        conv_text = ""
        #
        for utt in self.utterances:
            text = utt.text
            speaker = utt.get_speaker().id
            conv_text = conv_text + speaker + ": " + text + "\n\n"

        if mode_prompt == "rating":
            formatted_prompt = prompt_ratingscale.format(
                conv_text=conv_text, post=self.conv_topic
            )
        else:
            formatted_prompt = prompt_realnumber.format(
                conv_text=conv_text, post=self.conv_topic
            )
        annotations_ci = []
        try:
            response_text = prompt_gpt4(
                formatted_prompt, self.openaiKey, self.model_type, self.llm
            )
            # print(formatted_prompt)
            annotations_ci.append(response_text)
        except Exception as e:
            print("Error: ", e)
            annotations_ci.append(-1)
        return annotations_ci

# -*- coding: utf-8 -*-
import time

import openai

#################################################################################

prompt_realnumber = """{conv_text}'\n\n\n

The texts above show a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating 
inflammatory and aggressive speech.
Now, please use chain-of-thought reasoning to evaluate the average overall quality of the arguments presented in the above discussion.
Although a definition of overall argument quality does not exist so far, try to judge about the
overall quality based on all those argument quality dimensions that you think influence the overall quality of the given argumentation.
After the Chain-of-Thoughts reasoning steps, you should assign a score for the average overall argument quality of the arguments presented in the entire discussion on a scale from 1 to 5, where 1 is of poor quality and 5 is of extremely good quality. 
Conclude your evaluation with the statement: 'The average overall quality of the arguments presented in the above discussion is: [X]', where X is the numeric score (real number) you've determined. Please, ensure that your last statement is the score in brackets [].
"""
prompt_ratingscale = """{conv_text}'\n\n\n

The texts above show a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating 
inflammatory and aggressive speech.
Now, please use chain-of-thought reasoning to evaluate the average overall quality of the arguments presented in the above discussion.
Although a definition of overall argument quality does not exist so far, try to judge about the
overall quality based on all those argument quality dimensions that you think influence the overall quality of the given argumentation.
After the Chain-of-Thoughts reasoning steps, you should assign a label for the average overall argument quality of the arguments presented in the entire discussion on a scale from 1 to 3, where 1 is low quality, 2 is medium quality  and 3 is high quality. 
Conclude your evaluation with the statement: 'The average overall quality of the arguments presented in the above discussion is: [X]', where X is the label you've determined. Please, ensure that your last statement is the label in brackets [].
"""


class OAQuality:
    def __init__(self, utterances, conv_topic, openaiKey, mode):
        self.utterances = utterances
        self.conv_topic = conv_topic
        self.openaiKey = openaiKey
        self.mode = mode

    def prompt_gpt4(self, prompt):
        openai.api_key = self.openaiKey

        ok = False
        counter = 0
        while (
            not ok
        ):  # to avoid "ServiceUnavailableError: The server is overloaded or not ready yet."
            counter = counter + 1
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0,
                )
                ok = True
            except Exception as ex:
                print("error", ex)
                print("sleep for 5 seconds")
                time.sleep(5)
                if counter > 10:
                    return -1
        return response["choices"][0]["message"]["content"]

    def calculate_ovargquality_scores(self):
        mode_prompt = self.mode
        conv_text = ""
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
            response_text = self.prompt_gpt4(formatted_prompt)
            #            print(formatted_prompt)
            annotations_ci.append(response_text)
        except Exception as e:  # happens for one instance
            print("Error: ", e)
            annotations_ci.append(-1)
        return annotations_ci

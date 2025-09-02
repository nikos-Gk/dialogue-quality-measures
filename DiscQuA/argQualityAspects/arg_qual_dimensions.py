# -*- coding: utf-8 -*-
from tqdm import tqdm

from DiscQuA.utils import dprint, prompt_gpt4

#######################################################################################################################################
ini = """Below is given a set of definitions of various argument quality dimensions."""
#######################################################################################################################################

logic_prompt = """\n\n
Logic argument quality dimensions:

Label 1a: 
Local Acceptability: A premise of an argument should be seen as acceptable if it is worthy of being believed, i.e., if you rationally think it is true or if you see no reason for not believing that it may be true.
If you identify more than one premise in the comment, try to adequately weight the acceptability of each premise when judging about their “aggregate” acceptability—unless there are particular premises that dominate
your view of the author’s argumentation.    

Label 1b: 
Local Relevance: A premise of an argument should be seen as relevant if it contributes to the acceptance or rejection of the argument’s conclusion, i.e., if you think it is worthy of being considered as a reason, evidence, or similar regarding the conclusion. 
If you identify more than one premise in the comment, try to adequately weight the relevance of each premise when judging about their “aggregate” relevance—unless there are particular premises that dominate your view of the author’s argumentation. 
You should be open to see a premise as relevant even if it does not match your own stance on the issue.

Label 1c: 
Local Sufficiency: The premises of an argument should be seen as sufficient if, together, they provide enough support to make it rational to draw the argument’s conclusion.
If you identify more than one conclusion in the comment, try to adequately weight the sufficiency of the premises for each conclusion when judging about their “aggregate” sufficiency—unless there are particular
premises or conclusions that dominate your view of the author’s argumentation.
Notice that you may see premises as sufficient even though you do not personally accept all of them, i.e., sufficiency does not presuppose acceptability.

Label 1_overall:
Cogency: An argument should be seen as cogent if it has individually acceptable premises that are relevant to the argument’s conclusion and that are sufficient to draw the conclusion.
Try to adequately weight your judgments about local acceptability, local relevance, and local sufficiency when judging about cogency—unless there is a particular dimension among these that dominates your
view of an argument. Accordingly, if you identify more than one argument, try to adequately weight the cogency of each argument when judging about their “aggregate” cogency—unless there is a particular
argument that dominates your view of the author’s argumentation.
"""
rhetoric_prompt = """\n\n
Rhetoric argument quality dimensions:

Label 2a:
Credibility: An argumentation should be seen as successful in creating credibility if it conveys arguments and other information in a way that makes the author worthy of credence, e.g., by indicating the honesty of
the author or by revealing the author’s knowledge or expertise regarding the discussed issue. It should be seen as not successful if rather the opposite holds. Decide in dubio pro reo, i.e., if you have no doubt about the author’s credibility, then do not judge him or
her to be not credible.

Label 2b:
Emotional Appeal: An argumentation should be seen as successful in making an emotional appeal if it conveys arguments and other information in a way that creates emotions, which make the target audience more open to the author’s arguments. It should be seen as not successful if rather the opposite holds.
Notice that you should not judge about the persuasive effect of the author’s argumentation, but you should decide whether the argumentation makes the target audience willing/unwilling to be persuaded by the author (or to agree/disagree with the author) in principle—or neither.

Label 2c:
Clarity: The style of an argumentation should be seen as clear if it uses gramatically correct and widely unambiguous language as well as if it avoids unnecessary complexity and deviation from the discussed issue. 
The used language should make it easy for you to understand without doubts what the author argues for and how.

Label 2d:
Appropriateness: The style of an argumentation should be seen as appropriate if the used language supports the creation of credibility and emotions as well as if it is proportional to the discussed issue.
The choice of words and the grammatical complexity should, in your view, appear suitable for the discussed issue within the given setting (discussion in an online chatroom), matching with how credibility and emotions are created via the content of the argumentation.

Label 2e:
Arrangement: An argumentation should be seen as well-arranged if it presents the given topic, the composed arguments, and its conclusion in the right order.
Usually, the general topic and the particularly discussed topics should be clear before arguing and concluding about them. 
Notice, however, that other orderings may be used on purpose and may still be suitable to achieve persuasion. Besides, notice that, within the given setting (conversation in an online chatroom),some parts may be clear (e.g., the main topic) and thus left implicit.

Label 2_overall:
Effectiveness: An argumentation should be seen as effective if it achieves to persuade you of the author’s stance on the discussed topic or—in case you already agreed with the stance before—if it corroborates your agreement with the stance. 
Besides the actual arguments, also take into consideration the credibility and the emotional force of the argumentation. Decide in dubio pro reo, i.e., if you have no doubt about the correctness of the author’s arguments, then do
not judge him or her to be not effective—unless you explicitly think that the arguments do not support the author’s stance.
"""
dialectic_prompt = """\n\n
Dialectic argument quality dimensions:

Label 3a:
Global Acceptability: An argumentation should be seen as globally acceptable if everyone from the expected target audience would accept both the consideration of the stated arguments within the discussion of the given issue and the way they are stated.
Notice that you may see an argumentation as globally acceptable even though the stated arguments do not persuade you of the author’s stance.

Label 3b:
Global Relevance: An argumentation should be seen as globally relevant if it contributes to the resolution of the given issue, i.e., if it provides arguments and/or other information that help to arrive at an ultimate conclusion regarding the discussed issue.
You should be open to see an argumentation as relevant even if it does not your match your stance on the issue. Rather, the question is whether the provided arguments and information are worthy of being considered within the discussion of the issue.

Label 3c:
Global Sufficiency: An argumentation should be seen as globally sufficient if it adequately rebuts those counter-arguments to its conclusion that can be anticipated.
Notice that it is not generally clear which and how many counter-arguments can be anticipated. There may be cases where it is infeasible to rebut all such objections. Please judge about global sufficiency according
to whether all main objections of an argumentation that you see are rebutted.

Label 3_overall:
Reasonableness: An argumentation should be seen as reasonable if it contributes to the resolution of the given topic in a sufficient way that is acceptable to everyone from the expected target audience.
Try to adequately weight your judgments about global acceptability, global relevance, and global sufficiency when judging about reasonableness—unless there is a particular dimension among these that dominates your view of the author’s argumentation. 
In doubt, give more credit to global acceptability and global relevance than to global sufficiency due to the limited feasibility of the latter.
"""
overall_prompt = """\n\n
Overall argument quality:

Label 4:
Overall quality: overall quality of the presented arguments.
Judge about the overall quality based on the argument quality dimensions that you think influence the overall quality of the given argumentation.
If there is anything not covered by these dimensions that influences your view of the author's argumentation, also take that into account.
"""
#######################################################################################################################################
final = """\n\n
You will presented with a conversation history (which can be empty if the new utterance is the first utterance made in the conversation) from a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}

Given the conversation history, please identify the arguments presented in each new utterance (identified by a unique user_id).
An argument can be seen as combination of a conclusion (in terms of a claim) and a set of premises (in terms of supporting reasons or evidence for the claim). However, parts of an argument may be implicit or may simply be missing. 

*CONVERSATION HISTORY*: "{conv_history}"

*NEW UTTERANCE*: "{utterance}"

Noteworthy, the conversation history is provided for you to simply understand the arguments made before the new utterance so as to help you better annotate the various argument quality dimensions in the new utterance.
Thus, please do not annotate the entire conversation but annotate only the argument(s) in the new utterance by determining an appropritate score for each argument quality dimension.
Please provide the final answer directly with no reasoning steps.
Ensure that your final answer clearly assigns a score for each argument quality dimension, on a scale from 1 to 3, where 1 is of low quality, 2 of medium quality and 3 of high quality. 
For clarity, your response should succinctly be presented in a structured list format, indicating the score (integer) for each argument quality dimensions as follows:

"""

final_overall = """\n\n
You will presented with a conversation history (which can be empty if the new utterance is the first utterance made in the conversation) from a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
Given the conversation history, please identify the arguments presented in each new utterance (identified by a unique user_id).
An argument can be seen as combination of a conclusion (in terms of a claim) and a set of premises (in terms of supporting reasons or evidence for the claim). However, parts of an argument may be implicit or may simply be missing. 

*CONVERSATION HISTORY*: "{conv_history}"

*NEW UTTERANCE*: "{utterance}"

You are an argument quality evaluator. 
Rate the overall quality of the arguments in the new utterance on a scale from 1 to 3, where 1 is of low quality, 2 of medium quality and 3 of high quality. 
Conclude your evaluation with the statement: 'The overall argument quality is: [X]', where X is the numeric score (integer) you've determined. 
Please, provide the final answer directly with no reasoning steps.
Please, ensure that your last statement is the score in brackets [].
"""

logic_output_format = """
-Label 1a: [1/2/3]
-Label 1b: [1/2/3]
-Label 1c: [1/2/3]
-Label 1_overall: [1/2/3]
"""

rhetoric_output_format = """
-Label 2a: [1/2/3]
-Label 2b: [1/2/3]
-Label 2c: [1/2/3]
-Label 2d: [1/2/3]
-Label 2e: [1/2/3]
-Label 2_overall: [1/2/3]
"""

dialectic_output_format = """
-Label 3a: [1/2/3]
-Label 3b: [1/2/3]
-Label 3c: [1/2/3]
-Label 3_overall: [1/2/3]
"""


class AQualityDimensions:
    def __init__(
        self, utterances, conv_topic, openaiKey, model_type, llm, ctx, dimension
    ):
        self.utterances = utterances
        self.conv_topic = conv_topic
        self.openaiKey = openaiKey
        self.model_type = model_type
        self.llm = llm
        self.ctx = ctx
        self.dimension = dimension

    def argquality_dimensions_scores(self):
        prompt = ""
        if self.dimension == "logic":
            prompt = ini + logic_prompt + final + logic_output_format
        elif self.dimension == "rhetoric":
            prompt = ini + rhetoric_prompt + final + rhetoric_output_format
        elif self.dimension == "dialectic":
            prompt = ini + dialectic_prompt + final + dialectic_output_format
        elif self.dimension == "overall":
            prompt = ini + overall_prompt + final_overall
        else:
            dprint("error", "No matching dimension")
            return []
        # prompt = ini + Argument_quality_dimensions + final
        annotations_ci = []
        #
        for index, utt in enumerate(
            tqdm(self.utterances, desc="Processing utterances")
        ):
            conv_hist = ""
            text = utt.text
            speaker = utt.get_speaker().id
            if index > 0:
                start_ctx = max(0, index - self.ctx)
                for i in range(start_ctx, index):
                    prev_utt = self.utterances[i]
                    prev_speaker = prev_utt.get_speaker().id
                    prev_text = prev_utt.text
                    conv_hist += f"\n<user_name={prev_speaker}>\n{prev_text}\n"
            #
            try:
                formatted_prompt = prompt.format(
                    utterance="<user_name=" + speaker + ">" + "\n" + text,
                    conv_history=conv_hist,
                    post=self.conv_topic,
                )
                response_text = prompt_gpt4(
                    formatted_prompt, self.openaiKey, self.model_type, self.llm
                )
                # print(formatted_prompt)
                annotations_ci.append(response_text)
            except Exception as e:
                print("Error: ", e)
                annotations_ci.append(-1)
        return annotations_ci

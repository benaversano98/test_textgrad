from textgrad.engine import EngineLM, get_engine
from textgrad.variable import Variable
from typing import List, Union
from textgrad.autograd import LLMCall, FormattedLLMCall
from textgrad.autograd import Module

class MultiFieldEvaluation(Module):
    def __init__(
        self,
        evaluation_instruction: Variable,
        role_descriptions: List[str],
        engine: Union[EngineLM, str] = None,
        system_prompt: Variable = None,
    ):
        """A module to compare two variables using a language model.

        :param evaluation_instruction: Instruction to use as prefix for the comparison, specifying the nature of the comparison.
        :type evaluation_instruction: Variable
        :param engine: The language model to use for the comparison.
        :type engine: EngineLM
        :param v1_role_description: Role description for the first variable, defaults to "prediction to evaluate"
        :type v1_role_description: str, optional
        :param v2_role_description: Role description for the second variable, defaults to "correct result"
        :type v2_role_description: str, optional
        :param system_prompt: System prompt to use for the comparison, defaults to "You are an evaluation system that compares two variables."
        :type system_prompt: Variable, optional
        
        :example:
        TODO: Add an example
        """
        super().__init__()
        self.evaluation_instruction = evaluation_instruction
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        self.role_descriptions = role_descriptions
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = Variable("You are an evaluation system that compares two variables.",
                                            requires_grad=False,
                                            role_description="system prompt for the evaluation")
        format_string_items = ["{{instruction}}"]
        for role_description in role_descriptions:
            format_string_items.append(f"**{role_description}**: {{{role_description}}}")
        format_string = "\n".join(format_string_items)
        self.format_string = format_string.format(instruction=self.evaluation_instruction, **{role_description: "{"+role_description+"}" for role_description in role_descriptions})
        self.fields = {"instruction": self.evaluation_instruction, **{role_description: None for role_description in role_descriptions}}
        self.formatted_llm_call = FormattedLLMCall(engine=self.engine,
                                                   format_string=self.format_string,
                                                   fields=self.fields,
                                                   system_prompt=self.system_prompt)

    def forward(self, inputs: List[Variable]):
        for role_description, var in zip(self.role_descriptions, inputs):
            var.set_role_description(role_description)
        inputs_call = {"instruction": self.evaluation_instruction, 
                       **{role_description: var for role_description, var in zip(self.role_descriptions, inputs)}}
        return self.formatted_llm_call(inputs=inputs_call,
                                       response_role_description=f"evaluation of the a prediction")


class MultiFieldTokenParsedEvaluation(MultiFieldEvaluation):
    def __init__(
        self,
        evaluation_instruction: Variable,
        role_descriptions: List[str],
        engine: Union[EngineLM, str] = None,
        system_prompt: Variable = None,
        parse_tags: List[str] = None
    ):
        super().__init__(evaluation_instruction, role_descriptions, engine, system_prompt=system_prompt)
        self.parse_tags = parse_tags
    
    def parse_output(self, response: Variable) -> str:
        """
        Parses the output response and returns the parsed response.

        :param response: The response to be parsed.
        :type response: Variable
        :return: The parsed response.
        :rtype: str
        """
        response_text = response.value
        response = response_text.split(self.parse_tags[0])[1].split(self.parse_tags[1])[0].strip()
        return response
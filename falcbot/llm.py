import os
from asapdiscovery.ml.models import ASAPMLModelRegistry

from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI


from pydantic import BaseModel, Field, validator

class ASAPMLModelQuery(BaseModel):
    """
    Model that defines the smiles string, biological target and property of interest
    """
    SMILES: str = Field(..., description="SMILES string of the query compound")
    biological_target: str = Field(..., description="Biological target for the compound")
    property: str = Field(..., description="Measured property for the compound")


class IsMLQuery(BaseModel):
    """
    Model that checks if a query is a machine learning query
    """
    value: bool = Field(..., description="Boolean value indicating if the query is a machine learning query")


def _and_join(lst):
    return " and ".join(lst)

_base_ml_prompt_template = "You are an expert scientist, parse the following making sure all SMILES strings are represented exactly as in the input: Be very careful and use only SMILES already in the prompt. Allowed variables for target are {targets} and for property are {properties} : {query}"

def _make_ml_prompt_template() -> PromptTemplate:
    """
    Create a prompt template for the ASAPMLModelQuery model
    """
    # join to make a string with "and" between each
    targets_w_models = ASAPMLModelRegistry.get_targets_with_models()
    # filter out None values
    targets_w_models = [t for t in targets_w_models if t is not None]        
    target_str = _and_join(targets_w_models)
    properties = _and_join(ASAPMLModelRegistry.get_endpoints())
    pt = PromptTemplate(_base_ml_prompt_template)
    formatted =  pt.partial_format(targets=target_str, properties=properties)
    return formatted

_ML_PROMPT_TEMPLATE = _make_ml_prompt_template()



_base_is_query_prompt_template = "You are an expert scientist, parse the following and determine if it is a request for a prediction from a machine learning model, look for words like predict, : {query}"
_IS_ML_QUERY_PROMPT_TEMPLATE = PromptTemplate(_base_is_query_prompt_template)



class StructuredLLMQuery:

    def __init__(self, pydantic_model: BaseModel, prompt_template: PromptTemplate,  openai_model="gpt-4o",):
        """
        """
        self.openai_model = openai_model
        self.pydantic_model = pydantic_model
        self.prompt_template = prompt_template
        # get openai api key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        llm = OpenAI(model=self.openai_model)

        self.program = LLMTextCompletionProgram.from_defaults(
            output_cls=self.pydantic_model,
            prompt=self.prompt_template,
            llm=llm,
            verbose=True)


    def query(self, query: str):
        try:
            parsed_model = self.program(query=query)
            return True, parsed_model
        
        except Exception as e:
            print(e)
            return False, None

_BASIC_ML_LLM = StructuredLLMQuery(ASAPMLModelQuery, _ML_PROMPT_TEMPLATE)

_IS_ML_QUERY_LLM = StructuredLLMQuery(IsMLQuery, _IS_ML_QUERY_PROMPT_TEMPLATE)
import os
from asapdiscovery.ml.models import ASAPMLModelRegistry
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags

import util 
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core import PromptTemplate


from pydantic import BaseModel, Field, validator

class ASAPMLModelQuery(BaseModel):
    """
    Model that defines the smiles string, biological target and property of interest
    """
    SMILES: str = Field(..., description="SMILES string of the query compound")
    biological_target: str = Field(..., description="Biological target for the compound")
    property: str = Field(..., description="Measured property for the compound")


    # VALIDATE IN slack function to give feedback?

    # @validator("SMILES")
    # @classmethod
    # def validate_smiles(cls, v):
    #     if not _is_valid_smiles(v):
    #         raise ValueError("Invalid SMILES string")
    #     return v
    
    # @validator("biological_target")
    # @classmethod
    # def validate_target(cls, v):
    #     if v not in TargetTags.get_values():
    #         raise ValueError("Invalid target")
    #     return v
    
    # @validator("property")
    # @classmethod
    # def validate_property(cls, v):
    #     if v not in ASAPMLModelRegistry.get_endpoints():
    #         raise ValueError("Invalid property")
    #     return v
    

def _and_join(lst):
    return " and ".join(lst)

_base_ml_prompt_template = "You are an expert scientist, parse the following making sure all SMILES strings are represented exactly as in the input: Be very careful and use only SMILES already in the prompt. Allowed variables for target are {targets} and for property are {properties} : {query}"

def _make_ml_prompt_template() -> PromptTemplate:
    """
    Create a prompt template for the ASAPMLModelQuery model
    """
    # join to make a string with "and" between each
    targets_w_models = ASAPMLModelRegistry.get_targets_with_models()
    target_str = _and_join(targets_w_models)
    properties = _and_join(ASAPMLModelRegistry.get_endpoints())

    return _base_ml_prompt_template.partial_format(targets=target_str, properties=properties)

_ML_PROMPT_TEMPLATE = _make_ml_prompt_template()




class StructuredLLMQuery:

    def __init__(self, pydantic_model: BaseModel, prompt_template: str,  openai_model="gpt-4o",):
        """
        """
        self.openai_model = openai_model
        self.pydantic_model = pydantic_model
        self.prompt_template = prompt_template
        # get openai api key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.program = LLMTextCompletionProgram.from_defaults(
            output_cls=self.pydantic_model,
            prompt_template_str=self.prompt_template,
            verbose=True)


    def query(self, query: str):
        try:
            parsed_model = self.program(query=query)
            return True, parsed_model
        except Exception as e:
            print(e)
            return False, None

_BASIC_ML_LLM = StructuredLLMQuery(ASAPMLModelQuery, _ML_PROMPT_TEMPLATE)
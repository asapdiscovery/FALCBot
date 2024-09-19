import logging
import re
import uuid
import logging
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
from pydantic import BaseSettings, Field
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from alchemiscale import Scope
from openfe import ProteinComponent
from asapdiscovery.alchemy.schema.fec import (
    FreeEnergyCalculationFactory,
    AlchemiscaleSettings,
)
from asapdiscovery.alchemy.schema.prep_workflow import AlchemyPrepWorkflow
from asapdiscovery.alchemy.utils import AlchemiscaleHelper

from asapdiscovery.data.schema.complex import Complex, PreppedComplex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.backend.openeye import oechem
from asapdiscovery.data.services.postera.postera_factory import PosteraFactory
from asapdiscovery.data.services.services_config import CloudfrontSettings, S3Settings
from asapdiscovery.data.services.aws.cloudfront import CloudFront
from asapdiscovery.data.services.aws.s3 import S3

from asapdiscovery.ml.inference import GATInference, SchnetInference
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.ml.models import ASAPMLModelRegistry
import llm
import util

# from falcbot.sqlite_db import connect_sqlite_db, insert_series, create_series_table

from rdkit import Chem
import sqlite3

from multiprocessing import cpu_count

# logger in a global context
logging.basicConfig(level=logging.DEBUG)


class SlackSettings(BaseSettings):
    SLACK_BOT_TOKEN: str = Field(
        description="The Slack bot token.", env="SLACK_BOT_TOKEN"
    )
    SLACK_APP_TOKEN: str = Field(
        description="The Slack app token.", env="SLACK_APP_TOKEN"
    )


settings = SlackSettings()
app = App(token=settings.SLACK_BOT_TOKEN)



def are_you_alive_matcher(event, logger, context):
    # regex for any instance of help, case insensitive with optional spaces
    msg = event.get("text", None)
    if not msg:
        return False
    pattern = r"(?i)are you alive"
    match = re.search(pattern, msg)
    return match


@app.event("app_mention", matchers=[are_you_alive_matcher])
def are_you_alive(event, say, context, logger):
    say(f"yes im alive!")



def pred_matcher(event, logger, context):
    # regex for any instance of help, case insensitive with optional spaces
    msg = event.get("text", None)
    if not msg:
        return False
    worked, model = llm._IS_ML_QUERY_LLM.query(msg)
    if not worked:
        return False
    if worked:
        return model.value


@app.event("app_mention", matchers=[pred_matcher])
def make_pic50_pred(event, say, context, logger):
    content = event.get("text")
    # parse with LLM
    worked, model = llm._BASIC_ML_LLM.query(content)
    if not worked:
        say("Failed to parse the message, try something like `I would like to predict pIC50 for compound CCCC for MERS`")
        return
    
    # get the SMILES, target and property as parsed by the LLM
    smiles = model.SMILES
    target = model.biological_target
    endpoint = model.property # llm found property better


    if not util._is_valid_smiles(smiles):
        say(f"Invalid SMILES {smiles}, unable to proceed")
        return
    
    if not target in ASAPMLModelRegistry.get_targets_with_models():
        say(
            f"Invalid target {target}, not in: {ASAPMLModelRegistry.get_targets_with_models()}; unable to proceed"
        )
        return
    
    if not endpoint in ASAPMLModelRegistry.get_endpoints():
        say(
            f"Invalid endpoint {endpoint}, not in: {ASAPMLModelRegistry.get_endpoints()}; unable to proceed"
        )
        return
    
    _global_model = False

    if not ASAPMLModelRegistry.endpoint_has_target(endpoint):
        _target = None
        _global_model = True
        _target_str = "global"
    else:
        _target = target
        _target_str = target

    
    
    smiles = util._rdkit_smiles_roundtrip(smiles)
    model = ASAPMLModelRegistry.get_latest_model_for_target_type_and_endpoint(_target, "GAT", endpoint)
    if model is None:
        say(f"No model found for {target} {endpoint}")
        return
    infr = GATInference.from_ml_model_spec(model)
    pred = infr.predict_from_smiles(smiles)
    say(
        f"Predicted {_target_str} {endpoint} for {smiles} is {pred:.2f} using model {infr.model_name} :test_tube:" + (" (global model)" if _global_model else "")
    )
    # TODO make pred for every target if none specified

    return 



def list_targets_matcher(event, logger, context):
    # regex for any instance of help, case insensitive with optional spaces
    msg = event.get("text", None)
    if not event:
        return False
    pattern = r"(?i)list valid targets"
    match = re.search(pattern, msg)
    return match


@app.event("app_mention", matchers=[list_targets_matcher])
def list_all_targets(say, context, logger):
    targets = ASAPMLModelRegistry.get_targets_with_models()
    # filter out None values
    targets = [t for t in targets if t is not None]
    say(f"Targets: {targets}")
    return


def list_endpoints_matcher(event, logger, context):
    # regex for any instance of help, case insensitive with optional spaces
    msg = event.get("text", None)
    if not event:
        return False
    pattern = r"(?i)list valid endpoints"
    match = re.search(pattern, msg)
    return match


@app.event("app_mention", matchers=[list_endpoints_matcher])
def list_endpoints(say, context, logger):
    say(f"Endpoints: {ASAPMLModelRegistry.get_endpoints()}")
    return


def help_matcher(event, logger, context):
    # regex for any instance of help, case insensitive with optional spaces
    msg = event.get("text", None)
    if not event:
        return True
    pattern = r"(?i)help"
    match = re.search(pattern, msg)
    return match



@app.event("app_mention", matchers=[help_matcher])
def help_with_msg(say, context, event, logger):
    help(say, context, event, logger)


@app.event("app_mention")
def help_on_mention(say, context, event, logger):
    help(say, context, event, logger)


def help(say, context, event, logger):
    say(
        "you asked for help or misspelt a command, I can help you with the following commands:\n"
    )
    say("* `@falcbot predict <endpoint> for compound <smiles> for <target>`")
    say("* `@falcbot list valid targets`")
    say("* `@falcbot list valid endpoints`")
    say("* `@falcbot are you alive`")
    say("* `@falcbot help`")


@app.event("message")
def base_handle_message_events(body, logger):
    logger.debug(body)


# Start app
if __name__ == "__main__":
    SocketModeHandler(app, settings.SLACK_APP_TOKEN).start()

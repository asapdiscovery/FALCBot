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


def connect_sqlite_db(path, check_same_thread=False):
    connection = None
    try:
        connection = sqlite3.connect(path, check_same_thread=check_same_thread)
        print("Connection to SQLite DB successful")
    except sqlite3.Error as e:
        print(f"The error '{e}' occurred")

    return connection


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except sqlite3.Error as e:
        print(f"The error '{e}' occurred")


def create_series_table(connection):
    create_series_table_query = """
    CREATE TABLE IF NOT EXISTS series (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        factory_url TEXT NOT NULL,
        planned_network_url TEXT NOT NULL,
        ligands_url TEXT NOT NULL,
        receptor_url TEXT NOT NULL

    );
    """
    execute_query(connection, create_series_table_query)


def insert_series(
    connection, name, factory_url, planned_network_url, ligands_url, receptor_url
):
    insert_series_query = f"""
    INSERT INTO series (name, factory_url, planned_network_url, ligands_url, receptor_url)
    VALUES ('{name}', '{factory_url}', '{planned_network_url}', '{ligands_url}', '{receptor_url}');
    """
    execute_query(connection, insert_series_query)


def query_series_by_name(connection, name):
    query = f"SELECT * FROM series WHERE name='{name}'"
    cursor = connection.cursor()
    cursor.execute(query)
    # unpack into a dictionary
    series = cursor.fetchone()
    print(series)
    return series


settings = SlackSettings()
app = App(token=settings.SLACK_BOT_TOKEN)
db_connection = connect_sqlite_db("falcbot.sqlite3", check_same_thread=False)
create_series_table(db_connection)


_status_keys = ["complete", "running", "waiting", "error", "invalid", "deleted"]


def _download_slack_file(file_url, file_name):
    import requests

    headers = {"Authorization": f"Bearer {settings.SLACK_BOT_TOKEN}"}

    response = requests.get(file_url, headers=headers, stream=True)
    response.raise_for_status()
    with open(file_name, "wb") as f:
        for chunk in response.iter_content(chunk_size=2048):
            f.write(chunk)


def _push_to_s3_with_cloudfront(
    s3_instance: S3,
    cloudfront_instance: CloudFront,
    bucket_path: str,
    file_path: str,
    expires_delta: timedelta = timedelta(days=365 * 5),
    content_type: str = "application/json",
) -> str:
    # push to s3
    s3_instance.push_file(file_path, location=bucket_path, content_type=content_type)
    # generate cloudfront url
    expiry = datetime.utcnow() + expires_delta
    return cloudfront_instance.generate_signed_url(bucket_path, expiry)


def _link_to_block_data(link, text):
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"<{link}|{text}>"},
    }




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
    pattern = r"(?i)predict"
    match = re.search(pattern, msg)
    return match


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
    
    
    smiles = util._rdkit_smiles_roundtrip(smiles)
    model = ASAPMLModelRegistry.get_latest_model_for_target_type_and_endpoint(target, "GAT", endpoint)
    infr = GATInference.from_ml_model_spec(model)
    pred = infr.predict_from_smiles(smiles)
    say(
        f"Predicted {target} {endpoint} for {smiles} is {pred:.2f} using model {infr.model_name} :test_tube:"
    )

    # TODO make pred for every target if none specified


def make_structural_pred_matcher(event, logger, context):
    # regex for any instance of help, case insensitive with optional spaces
    msg = event.get("text", None)
    if not msg:
        return False
    pattern = r"(?i)predict pIC50 for structure"
    match = re.search(pattern, msg)
    return match


@app.event("app_mention", matchers=[make_structural_pred_matcher])
def make_structural_pred(event, say, context, logger):
    content = event.get("text")
    # parse message for molset using regex
    pattern = r"(?i)\s+for\s+target\s+(.+)"
    match = re.search(pattern, content)
    if match:
        target = match.group(1)
    else:
        say("Could not find Target in the message, unable to proceed")
        return

    allowed_targets = list(
        set(ASAPMLModelRegistry.get_targets_with_models()) - {"SARS-CoV-2-Mac1"}
    )  # remove SARS-CoV-2-Mac1, currently not supported

    if not target in allowed_targets:
        say(f"Invalid target {target}, not in: {allowed_targets}; unable to proceed")
        return

    # check for attatched file
    files = event.get("files")
    if not files:
        logger.info("No file attatched, unable to proceed")
        say("No pdb file attatched, unable to proceed")
        return
    else:
        if len(files) > 1:
            logger.info("More than one file attatched, unable to proceed")
            say("More than one file attatched, unable to proceed")
            return
        # get the first file
        file = files[0]
        title = file.get("title")
        # check if it is a pdb file
        file_extn = file.get("title").split(".")[-1]
        if file_extn != "pdb":
            say("Attatched file is not a pdb file, unable to proceed")
            return

    try:
        with NamedTemporaryFile(suffix=".pdb") as temp:
            logger.info(f"file: {file.get('url_private_download')}")
            _download_slack_file(file.get("url_private_download"), temp.name)
            ref_complex = Complex.from_pdb(
                temp.name,
                target_kwargs={"target_name": f"receptor"},
                ligand_kwargs={"compound_name": f"receptor_ligand"},
            )
    except Exception as e:
        say(f"Failed to load receptor from attatched file with error: {e}")
        return

    # make prediction
    si = SchnetInference.from_latest_by_target(target)
    pred = si.predict_from_oemol(ref_complex.to_combined_oemol())
    say(
        f"Predicted pIC50 for {title} is {pred:.2f} using model {si.model_name} :test_tube:"
    )

    # TODO make pred for every target if none specified


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
    say(f"Targets: {ASAPMLModelRegistry.get_targets_with_models()}")
    return


def list_endpoints_matcher(event, logger, context):
    # regex for any instance of help, case insensitive with optional spaces
    msg = event.get("text", None)
    if not event:
        return False
    pattern = r"(?i)list valid endpoints"
    match = re.search(pattern, msg)
    return match


@app.event("app_mention", matchers=[list_targets_matcher])
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
    say("* `@falcbot predict pIC50 for structure for target <target>`")
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

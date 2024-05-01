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


def _is_valid_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    else:
        return True


def _rdkit_smiles_roundtrip(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol)


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


def query_all_networks_matcher(event, logger, context):
    # regex for any instance of help, case insensitive with optional spaces
    msg = event.get("text", None)
    if not msg:
        return False
    pattern = r"(?i)query all networks"
    match = re.search(pattern, msg)
    return match


@app.event("app_mention", matchers=[query_all_networks_matcher])
def query_all_networks(event, say, context, logger):
    logger.debug("Querying all networks")
    client = AlchemiscaleHelper()
    scope_status_dict = client._client.get_scope_status(visualize=False)
    for k, v in scope_status_dict.items():
        say(f"Status {k} has count {v}")

    say("________________________________")
    say("Checking for running networks...")

    running_networks = client._client.query_networks()

    if not running_networks:
        say("No networks are running currently")
        return

    networks_status = client._client.get_networks_status(running_networks)
    networks_actioned_tasks = client._client.get_networks_actioned_tasks(
        running_networks
    )

    for key, network_status, actioned_tasks in zip(
        running_networks, networks_status, networks_actioned_tasks
    ):
        if (
            "running" in network_status or "waiting" in network_status
        ) and actioned_tasks:
            say(f"Network {key} has following status breakdown")
            state_breakdown = ""
            for state in _status_keys:
                state_breakdown += f"{state}: {network_status.get(state, 0)} "
            say(state_breakdown)
            say("________________________________")
    say("Done :smile:")


def run_fec_matcher(event, logger, context):
    # regex for any instance of help, case insensitive with optional spaces
    msg = event.get("text", None)
    if not msg:
        return False
    pattern = r"(?i)run FEC"
    match = re.search(pattern, msg)
    return match


@app.event("app_mention", matchers=[run_fec_matcher])
def run_fec(event, say, context, logger):
    logger.info("Planning and submitting from postera")
    say(
        "Preparing your calculation, please wait this may take a while, ... :ghost: :ghost: :ghost:"
    )
    content = event.get("text")
    # parse message for molset using regex
    pattern = r"on series\s+.*?(\b[^\s]+\b)+"
    match = re.search(pattern, content)
    if match:
        postera_molset_name = match.group(1)
        logger.info(f"Postera molecule set name is {postera_molset_name}")
    else:
        say(
            "Could not find postera molecule set name in the message, unable to proceed"
        )
        return

    campaign = "confidential"

    # check for attatched file
    files = event.get("files")
    if not files:
        logger.info("No file attatched, unable to proceed")
        say("No receptor file attatched, unable to proceed")
        return
    else:
        if len(files) > 1:
            logger.info("More than one file attatched, unable to proceed")
            say("More than one file attatched, unable to proceed")
            return
        # get the first file
        file = files[0]
        # check if it is a pdb file
        file_extn = file.get("title").split(".")[-1]
        if file_extn != "pdb":
            say("Attatched file is not a pdb file, unable to proceed")
            return

    # load ligands from postera
    try:
        input_ligands = PosteraFactory(molecule_set_name=postera_molset_name).pull()
    except Exception as e:
        say(f"Failed to pull ligands from postera with error: {e}")
        return

    say(
        f"Input series has {len(input_ligands)} ligands, this may take a while to process. I'll let you know once its running. Please be patient :ghost: :ghost: :ghost:"
    )
    fixed_ligands = []
    # add hydrogens to ligands
    for ligand in input_ligands:
        mol = ligand.to_oemol()
        oechem.OEAddExplicitHydrogens(mol)
        fixed_ligands.append(Ligand.from_oemol(mol))
    input_ligands = fixed_ligands
    # create dataset name
    dataset_name = postera_molset_name.replace("-", "_") + "_" + "FALCBot"
    project = dataset_name

    # run prep workflow
    logger.info("Running prep workflow")

    prep_factory = AlchemyPrepWorkflow()

    # load receptor from attatched file
    # read into temp file
    # TODO move to pre-prepped PDBs hosted on the cloud instance and pull from there
    try:
        with NamedTemporaryFile(suffix=".pdb") as temp:
            logger.info(f"file: {file.get('url_private_download')}")
            _download_slack_file(file.get("url_private_download"), temp.name)
            ref_complex = Complex.from_pdb(
                temp.name,
                target_kwargs={"target_name": f"{dataset_name}_receptor"},
                ligand_kwargs={"compound_name": f"{dataset_name}_receptor_ligand"},
            )
    except Exception as e:
        say(f"Failed to load receptor from attatched file with error: {e}")
        return
    # prep the complex
    logger.info("Prepping complex")
    prepped_ref_complex = PreppedComplex.from_complex(ref_complex)

    import time

    logger.info("Creating alchemy dataset")
    processors = cpu_count() - 1
    logger.info(f"Using {processors} processors")
    start_time = time.time()
    alchemy_dataset = prep_factory.create_alchemy_dataset(
        dataset_name=dataset_name,
        ligands=input_ligands,
        reference_complex=prepped_ref_complex,
        processors=processors,
    )
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Time taken to create alchemy dataset: {execution_time} seconds")

    # check for failed ligands
    logger.info("Checking for failed ligands")
    if alchemy_dataset.failed_ligands:
        fails = sum([len(values) for values in alchemy_dataset.failed_ligands.values()])
        say(f"Failed to prep {fails} ligands")
        # add more detail

    # we have our working ligands
    posed_ligands = alchemy_dataset.posed_ligands

    # ok now onto  actual network creation
    logger.info("Creating factory and planned network")
    factory = FreeEnergyCalculationFactory()

    # create receptor
    # write to a temp pdb file and read back in
    with NamedTemporaryFile(suffix=".pdb") as fp:
        alchemy_dataset.reference_complex.target.to_pdb_file(fp.name)
        receptor = ProteinComponent.from_pdb_file(fp.name)

    # create factory
    logger.info("Planning network with factory and planned network")
    planned_network = factory.create_fec_dataset(
        dataset_name=dataset_name,
        receptor=receptor,
        ligands=posed_ligands,
        central_ligand=None,
        experimental_protocol=None,
    )

    # we want to return links to the factory and planned network
    # we do this through artifacts in a cloudfront exposed bucket
    cf = CloudFront.from_settings(CloudfrontSettings())
    s3 = S3.from_settings(S3Settings())

    # push factory to cloudfront exposed bucket
    factory_fname = f"fec_factory-{dataset_name}.json"
    factory_bucket_path = f"alchemy/{dataset_name}/{factory_fname}"
    with NamedTemporaryFile() as temp:
        factory.to_file(filename=temp.name)
        factory_cf_url = _push_to_s3_with_cloudfront(
            s3, cf, factory_bucket_path, temp.name, content_type="application/json"
        )

    planned_network_fname = f"planned_network-{dataset_name}.json"
    planned_network_bucket_path = f"alchemy/{dataset_name}/{planned_network_fname}"
    # push planned network to cloudfront exposed bucket
    with NamedTemporaryFile() as temp:
        planned_network.to_file(filename=temp.name)
        planned_network_cf_url = _push_to_s3_with_cloudfront(
            s3,
            cf,
            planned_network_bucket_path,
            temp.name,
            content_type="application/json",
        )

    ligands_fname = f"ligands-{dataset_name}.sdf"
    ligands_fname_bucket_path = f"alchemy/{dataset_name}/{ligands_fname}"
    # push planned network to cloudfront exposed bucket
    with NamedTemporaryFile(suffix=".sdf") as temp:
        alchemy_dataset.save_posed_ligands(temp.name)
        ligand_cf_url = _push_to_s3_with_cloudfront(
            s3,
            cf,
            ligands_fname_bucket_path,
            temp.name,
            content_type="text/plain",
        )

    receptor_fname = f"receptor-{dataset_name}.pdb"
    receptor_fname_bucket_path = f"alchemy/{dataset_name}/{receptor_fname}"
    # push planned network to cloudfront exposed bucket
    with NamedTemporaryFile(suffix=".pdb") as temp:
        alchemy_dataset.reference_complex.target.to_pdb_file(temp.name)
        receptor_cf_url = _push_to_s3_with_cloudfront(
            s3,
            cf,
            receptor_fname_bucket_path,
            temp.name,
            content_type="text/plain",
        )

    logger.info(f"Data set name: {dataset_name}")
    logger.info(f"Factory url: {factory_cf_url}")
    logger.info(f"Planned network url: {planned_network_cf_url}")
    logger.info(f"Ligands url: {ligand_cf_url}")
    logger.info(f"Receptor url: {receptor_cf_url}")

    # submit the network
    client = AlchemiscaleHelper()

    network_scope = Scope(org="asap", campaign=campaign, project=project)
    submitted_network = client.create_network(
        planned_network=planned_network, scope=network_scope
    )
    task_ids = client.action_network(
        planned_network=submitted_network, prioritize=False
    )
    logger.debug(
        f"Submitted network {submitted_network.results.network_key} with task ids {task_ids} to campaign {campaign} and project {project}."
    )
    # except Exception as e:
    #     say(f"Failed to submit network with error: {e}")
    #     return

    insert_series(
        db_connection,
        dataset_name,
        factory_cf_url,
        planned_network_cf_url,
        ligand_cf_url,
        receptor_cf_url,
    )

    say(
        f"Simulations are running! :rocket: :rocket: :rocket: Your project name is: {project}, to debug use `@falcbot debug series {dataset_name}`"
    )


def debug_series_matcher(event, logger, context):
    # regex for any instance of help, case insensitive with optional spaces
    msg = event.get("text", None)
    if not msg:
        return False
    pattern = r"(?i)debug series"
    match = re.search(pattern, msg)
    return match


@app.event("app_mention", matchers=[debug_series_matcher])
def debug_series(event, say, context, logger):
    message = event.get("text")
    pattern = r"series\s+.*?(\b[^\s]+\b)+"
    match = re.search(pattern, message)
    if match:
        series_name = match.group(1)
        logger.info(f"Series name is {series_name}")
    else:
        say("Could not find series name in the message, unable to proceed")
        return

    # query the database
    series = query_series_by_name(db_connection, series_name)
    if not series:
        say(f"Series {series_name} not found in the database, unable to proceed")
        return
    say(f"Series {series_name} found with values: {series}")

    ligand_cf_url = series[4]
    receptor_cf_url = series[5]
    factory_cf_url = series[2]
    planned_network_cf_url = series[3]

    # make block data from the links
    block_data = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Links to your debugging info :pill: :pill: :pill:",
            },
        },
        _link_to_block_data(ligand_cf_url, "Ligand SDF file"),
        _link_to_block_data(receptor_cf_url, "Receptor PDB file"),
        _link_to_block_data(factory_cf_url, "FECFactory JSON"),
        _link_to_block_data(planned_network_cf_url, "PlannedNetwork JSON"),
    ]

    say("Links to your debugging info:", blocks=block_data)

    return


def make_pic50_pred_matcher(event, logger, context):
    # regex for any instance of help, case insensitive with optional spaces
    msg = event.get("text", None)
    if not msg:
        return False
    pattern = r"(?i)predict pIC50 for SMILES"
    match = re.search(pattern, msg)
    return match


@app.event("app_mention", matchers=[make_pic50_pred_matcher])
def make_pic50_pred(event, say, context, logger):
    content = event.get("text")
    # parse message for molset using regex
    pattern = r"(?i)SMILES\s+(.+?)\s+for\s+target\s+(.+)"
    match = re.search(pattern, content)
    if match:
        smiles = match.group(1)
        target = match.group(2)
    else:
        say("Could not find SMILES and Target in the message, unable to proceed")
        return
    if not _is_valid_smiles(smiles):
        say(f"Invalid SMILES {smiles}, unable to proceed")
        return
    if not target in ASAPMLModelRegistry.get_targets_with_models():
        say(
            f"Invalid target {target}, not in: {ASAPMLModelRegistry.get_targets_with_models()}; unable to proceed"
        )
        return
    # make prediction
    smiles = _rdkit_smiles_roundtrip(smiles)
    gs = GATInference.from_latest_by_target(target)
    pred = gs.predict_from_smiles(smiles)
    say(
        f"Predicted pIC50 for {smiles} is {pred:.2f} using model {gs.model_name} :test_tube:"
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
        "you asked for help or misspelt a command, I can help you with the following commands:"
    )
    say("* `@falcbot run FEC on series <series_name>`")
    say("* `@falcbot predict pIC50 for SMILES <smiles> for target <target>`")
    say("* `@falcbot predict pIC50 for structure for target <target>`")
    say("* `@falcbot list valid targets`")
    say("* `@falcbot query all networks`")
    say("* `@falcbot debug series <series_name>`")
    say("* `@falcbot are you alive`")
    say("* `@falcbot help`")


@app.event("message")
def base_handle_message_events(body, logger):
    logger.debug(body)


# Start app
if __name__ == "__main__":
    SocketModeHandler(app, settings.SLACK_APP_TOKEN).start()

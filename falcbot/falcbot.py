import logging
import re
import uuid
import logging
import tempfile

from pydantic import BaseSettings, Field
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from alchemiscale import Scope
from openfe import ProteinComponent
from asapdiscovery.alchemy.schema.fec import (
    FreeEnergyCalculationFactory,
    AlchemiscaleSettings,
)
from asapdiscovery.alchemy.utils import AlchemiscaleHelper
from asapdiscovery.data.services.postera.postera_factory import PosteraFactory


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

_status_keys = ["complete", "running", "waiting", "error", "invalid", "deleted"]


def _download_slack_file(file_url, file_name):
    import requests

    headers = {"Authorization": f"Bearer {settings.SLACK_BOT_TOKEN}"}

    response = requests.get(file_url, headers=headers, stream=True)
    response.raise_for_status()
    with open(file_name, "wb") as f:
        for chunk in response.iter_content(chunk_size=2048):
            f.write(chunk)


@app.message(re.compile("(hi|hello|hey)"))
def say_hello_regex(say, context):
    # regular expression matches are inside of context.matches
    print(context)
    greeting = context["matches"][0]
    say(f"{greeting}, how are you?")


@app.message(re.compile("(.*)are you alive(.*)"))
def are_you_alive(say, context):
    say(f"yes im alive!")


@app.message(re.compile("(.*)query all networks(.*)"))
def query_all_networks(say, context, logger):
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
    else:
        for key in running_networks:
            # get status
            network_status = client._client.get_network_status(
                network=key, visualize=False
            )
            running_tasks = client._client.get_network_actioned_tasks(network=key)
            if "running" in network_status or "waiting" in network_status:
                say(f"Network {key} has following status breakdown")
                state_breakdown = ""
                for state in _status_keys:
                    state_breakdown += f"{state}: {network_status.get(state, 0)} "
                say(state_breakdown)
                say("________________________________")


@app.message("plan and submit from postera molecule set")
def plan_and_submit_postera(message, say, context, logger):
    logger.info("Planning and submitting from postera")

    content = message.get("text")
    # parse message for molset using regex
    pattern = r"from postera molecule set\s+.*?(\b[^\s]+\b)\s+to"
    match = re.search(pattern, content)
    if match:
        postera_molset_name = match.group(1)
        logger.info(f"Postera molecule set name is {postera_molset_name}")
    else:
        say(
            "Could not find postera molecule set name in the message, unable to proceed"
        )
        return

    pattern = r"to campaign\s+.*?(\b[^\s]+\b)\s+and project\s+.*?(\b[^\s]+\b)"
    match = re.search(pattern, content)
    if match:
        campaign = match.group(1)
        logger.info(f"Campaign is {campaign}")
        project = match.group(2)
        logger.info(f"Project is {project}")
    else:
        say("Could not find campaign and project in the message, unable to proceed")
        return

    # check we have both campaign and project
    if not campaign or not project:
        say("Could not find campaign and project in the message, unable to proceed")
        return

    # check we have a valid campaign

    if campaign not in ("public", "confidential"):
        say(
            "Invalid campaign, must be one of: (public, confidential) unable to proceed"
        )
        return

    # check for attatched file
    files = message.get("files")
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

    factory = FreeEnergyCalculationFactory()

    # load ligands from postera
    try:
        input_ligands = PosteraFactory(molecule_set_name=postera_molset_name).pull()
    except Exception as e:
        say(f"Failed to pull ligands from postera with error: {e}")
        return

    # load receptor from attatched file
    # read into temp file
    try:
        with tempfile.NamedTemporaryFile() as temp:
            logger.info(f"file: {file.get('url_private_download')}")
            _download_slack_file(file.get("url_private_download"), temp.name)
            receptor = ProteinComponent.from_pdb_file(temp.name)
    except Exception as e:
        say(f"Failed to load receptor from attatched file with error: {e}")
        return

    dataset_name = postera_molset_name + "_" + str(uuid.uuid4())

    planned_network = factory.create_fec_dataset(
        dataset_name=dataset_name,
        receptor=receptor,
        ligands=input_ligands,
        central_ligand=None,
        experimental_protocol=None,
    )

    # submit the network
    client = AlchemiscaleHelper()

    network_scope = Scope(org="asapdiscovery", campaign=campaign, project=project)

    submitted_network = client.create_network(
        planned_network=planned_network, scope=network_scope
    )
    task_ids = client.action_network(
        planned_network=submitted_network, prioritize=False
    )
    logger.debug(f"Submitted network {submitted_network} with task ids {task_ids}")
    say(
        f"Submitted network {submitted_network} with task ids {task_ids} to campaign {campaign} and project {project}."
    )


@app.message(re.compile("(.*)plan and submit from JSON(.*)"))
def plan_and_submit_json(say, context, logger): ...


@app.event("message")
def base_handle_message_events(body, logger):
    logger.info(body)


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, settings.SLACK_APP_TOKEN).start()

import logging
import re
import uuid
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

logger = logging.getLogger(__name__)


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
def query_all_networks(say, context):
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


@app.message(re.compile("(.*)plan and submit from postera molecule set(.*)"))
def plan_and_submit_postera(say, context):
    logger.debug("Planning and submitting from postera")
    # create
    postera_molset_name = "X"
    campaign = "Y"
    project = "P"
    exp_protocol = "XXXX"

    factory = FreeEnergyCalculationFactory()

    # load ligands from postera
    input_ligands = PosteraFactory(molset_name=postera_molset_name).load()

    # load receptor from attatched file

    receptor = ProteinComponent.from_pdb_file(receptor)

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
def plan_and_submit_json(say, context):
    ...


@app.event("message")
def base_handle_message_events(body, logger):
    logger.info(body)


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, settings.SLACK_APP_TOKEN).start()

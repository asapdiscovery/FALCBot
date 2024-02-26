import logging
import re

from pydantic import BaseSettings, Field
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

logger = logging.getLogger(__name__)


class SlackSettings(BaseSettings):
    SLACK_BOT_TOKEN: str = Field(description="The Slack bot token.", env="SLACK_BOT_TOKEN")
    SLACK_APP_TOKEN: str = Field(description="The Slack app token.", env="SLACK_APP_TOKEN")
    # SLACK_SIGNING_SECRET: str = Field(description="The Slack signing secret.")
    # SLACK_CHANNEL: str = Field(description="The Slack channel to post to.")

settings = SlackSettings()
app = App(token=settings.SLACK_BOT_TOKEN)


@app.message("hello")
def message_hello(message, say):
    logger.debug(f"Received a message: {message}")
    print("messgae received")
    # say() sends a message to the channel where the event was triggered
    say(f"Hey there <@{message['user']}>!")

# Start your app
if __name__ == "__main__":
        print(settings.SLACK_APP_TOKEN)
        print(settings.SLACK_BOT_TOKEN)
        SocketModeHandler(app, settings.SLACK_APP_TOKEN).start()



















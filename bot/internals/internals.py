import collections
import configparser
import logging
import pathlib
import os
import typing as tp

import aiogram
import aiogram.contrib.fsm_storage.memory
import ujson

async def set_commands(bot: aiogram.Bot) -> None:
    commands = [
        aiogram.types.bot_command.BotCommand("/start", description="Starts the bot and shows a greeting message."),
        aiogram.types.bot_command.BotCommand("/help", description="Show the help screen."),
        aiogram.types.bot_command.BotCommand("/content", description="Upload an image to be stylized."),
        aiogram.types.bot_command.BotCommand("/style", description="Select or upload an image to be used as style source."),
        aiogram.types.bot_command.BotCommand("/transfer", description="Perform style transfer."),
        aiogram.types.bot_command.BotCommand("/cancel", description="Cancel current command."),
    ]
    await bot.set_my_commands(commands)

CONFIG = configparser.ConfigParser()
ROOT = pathlib.Path(__file__).parent.parent.parent.resolve().absolute()
CONFIG.read((ROOT / "config" / "bot.ini"))
logging.basicConfig(level=logging.getLevelName(CONFIG["verbose"].get("verbosity", "ERROR")), format="""
[%(levelname)s] %(module)s:%(lineno)d at %(asctime)s, 
    logger: %(name)s — %(message)s""")
LOGGER = logging.getLogger(__name__)
BOT = aiogram.Bot(token=CONFIG["bot"]["token"], parse_mode="MarkdownV2")
DISPATCHER = aiogram.Dispatcher(BOT, storage=aiogram.contrib.fsm_storage.memory.MemoryStorage())

_styles_path = pathlib.Path(CONFIG["styles"]["config_path"])
if not _styles_path.is_absolute():
    _styles_path = (ROOT / _styles_path).resolve().absolute()
with open(_styles_path, "r") as styles_json:
    STYLES = ujson.load(styles_json)
for i in STYLES:
    STYLES[i]["path"] = pathlib.Path(STYLES[i]["path"]).resolve().absolute()
STYLES["custom"] = {
    "display_name": "Custom style image",
    "path": None
}
STYLES = collections.OrderedDict(sorted(STYLES.items(), key=lambda t: t[0]))

async def on_startup(dp: aiogram.Dispatcher) -> None:
    await set_commands(BOT)
    if CONFIG["bot"]["event_strategy"] == "webhook":
        LOGGER.warning("Setting up webhook.")
        if not CONFIG["bot"].get("webhook_host"):
            raise ValueError("Webhook host specification is incorrect. Please check your config.")
        if not CONFIG["bot"].get("webhook_path"):
            LOGGER.warn("Webhook path not provided, assuming we should call /")
            CONFIG["bot"]["webhook_path"] = "/"
        await BOT.set_webhook(str(CONFIG["bot"]["webhook_host"] + CONFIG["bot"]["webhook_path"]))

async def on_shutdown(dp: aiogram.Dispatcher) -> None:
    LOGGER.warning("Deleting webhook")
    await BOT.delete_webhook()
    LOGGER.warning("Shutting down...")

def start_bot() -> None:
    if CONFIG["bot"].get("event_strategy") == "polling":
        LOGGER.warning("Starting polling...")
        aiogram.utils.executor.start_polling(DISPATCHER, skip_updates=True, on_startup=on_startup)
    elif CONFIG["bot"].get("event_strategy") == "webhook":
        if not (CONFIG["bot"]["webhook_path"] and CONFIG["bot"]["webhook_host"]):
            raise ValueError("\"webhook\" strategy selected but no host provided.")
        aiogram.utils.executor.start_webhook(dispatcher=DISPATCHER,
            webhook_path=CONFIG["bot"]["webhook_path"],
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            skip_updates=True,
            host="0.0.0.0",
            port=int(os.getenv("PORT")) # heroku assigns a random open port to each dyno, so assume we have an envvar denoting it.
        )
    else:
        raise ValueError("Launch strategy incorrect. Please check your config for correctness.")
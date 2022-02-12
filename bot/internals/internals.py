import configparser
import logging
import pathlib
import os
import typing as tp

import aiogram

CONFIG = configparser.ConfigParser()
CONFIG.read((pathlib.Path(__file__).parent.parent.parent.resolve() / "config" / "bot.ini"))
CONFIG["styles"]["style_paths"] = _create_style_paths()
logging.basicConfig(level=logging.getLevelName(CONFIG["verbose"].get("verbosity", "ERROR")), format="""
[%(levelname)s] %(module)s:%(lineno)d at %(asctime)s, 
    logger: %(name)s — %(message)s""")
LOGGER = logging.getLogger(__name__)
BOT = aiogram.Bot(token=CONFIG["bot"]["token"], parse_mode="MarkdownV2")
DISPATCHER = aiogram.Dispatcher(BOT)

def _create_style_paths() -> tp.Dict[str, tp.Optional[pathlib.Path]]:
    style_dir = pathlib.Path(__file__).parent.parent.parent.resolve() / styles
    if not style_dir.is_dir():
        style_paths = {
            "custom": None
        }
        return style_paths
    files = [pathlib.Path(i).resolve() for i in CONFIG["paths"]["style_files"].split(",")]
    names = CONFIG["styles"]["style_names"].split(",")
    if len(files) != len(names):
        raise ValueError("Incorrect config: number of style paths does not match the number of style names.")
    style_paths = dict(zip(names, files))
    style_paths["custom"] = None
    return style_paths

async def on_startup(dp: aiogram.Dispatcher) -> None:
    LOGGER.warning("Setting up webhook")
    if not (CONFIG["bot"].get("webhook_host") or CONFIG["bot"].get("webhook_port")):
        raise ValueError("Webhook host or port specification is incorrect. Please check your config.")
    if not CONFIG["bot"].get("webhook_path"):
        LOGGER.warn("Webhook path not provided, assuming we should call /")
        CONFIG["bot"]["webhook_path"] = "/"
    await BOT.set_webhook(CONFIG["bot"]["webhook_path"])

async def on_shutdown(dp: aiogram.Dispatcher) -> None:
    LOGGER.warning("Deleting webhook")
    await BOT.delete_webhook()
    LOGGER.warning("Shutting down...")

def start_bot() -> None:
    if CONFIG["bot"].get("event_strategy") == "POLLING":
        LOGGER.warning("Starting polling...")
        aiogram.executor.start_polling(DISPATCHER, skip_updates=True)
    elif CONFIG["bot"].get("event_strategy") == "WEBHOOK":
        if not (CONFIG["bot"]["webhook_path"] and CONFIG["bot"]["webhook_host"] and CONFIG["bot"]["webhook_port"]):
            raise ValueError("WEBHOOK strategy selected but not all required parameters are provided.")
        aiogram.utils.executor.start_webhook(dispatcher=DISPATCHER,
            webhook_path=CONFIG["bot"]["webhook_path"],
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            skip_updates=True,
            host=CONFIG["bot"]["webhook_host"],
            port=CONFIG["bot"]["webhook_host"]
        )
    else:
        raise ValueError("Launch strategy incorrect. Please check your .env file for correctness.")
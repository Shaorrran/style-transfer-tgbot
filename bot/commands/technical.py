import logging

from bot import internals

import aiogram

LOGGER = logging.getLogger(__name__)

@internals.DISPATCHER.message_handler(commands=["start"])
async def start(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext) -> None:
    LOGGER.debug("/start detected.")
    await state.finish()
    await message.reply(aiogram.utils.markdown.text(
        aiogram.utils.markdown.text("Hello\!"),
        aiogram.utils.markdown.text("This bot uses neural networks to create stylized images from a source image and a style sample image\."),
        aiogram.utils.markdown.text(
            aiogram.utils.markdown.text("Please run"),
            aiogram.utils.markdown.code("/help"),
            aiogram.utils.markdown.text("to learn more\."),
            sep=" "
        ),
        sep="\n"
    ))

@internals.DISPATCHER.message_handler(commands=["help"])
async def help(message: aiogram.types.Message) -> None:
    LOGGER.debug("Serving /help.")
    await message.reply(aiogram.utils.markdown.text(
        aiogram.utils.markdown.text("These are the commands available to you\:"),
        aiogram.utils.markdown.text(
            aiogram.utils.markdown.code("/start"),
            aiogram.utils.markdown.text("Starts the bot\."),
            sep=" — "
        ),
        aiogram.utils.markdown.text(
            aiogram.utils.markdown.code("/help"),
            aiogram.utils.markdown.text("Prints this help message\."),
            sep=" — "
        ),
        aiogram.utils.markdown.text(
            aiogram.utils.markdown.code("/content"),
            aiogram.utils.markdown.text(" or "),
            aiogram.utils.markdown.code("/image"),
            aiogram.utils.markdown.text(" or just uploading an image "),
            aiogram.utils.markdown.text("— Selects an image to be stylized\."),
            sep=""
        ),
        aiogram.utils.markdown.text(
            aiogram.utils.markdown.code("/style"),
            aiogram.utils.markdown.text("Lists all available style presets and ask you to choose one \(or upload your own\!\)\."),
            sep=" — "
        ),
        aiogram.utils.markdown.text(
            aiogram.utils.markdown.code("/transfer"),
            aiogram.utils.markdown.text(" or "),
            aiogram.utils.markdown.code("/run"),
            aiogram.utils.markdown.text(" — Start style transfer processing \(this may take some time\)\."),
            sep=""
        ),
        sep="\n"
    ))

@internals.DISPATCHER.message_handler(state="*", commands="cancel")
@internals.DISPATCHER.message_handler(aiogram.dispatcher.filters.Text(equals="cancel", ignore_case=True), state="*")
@internals.DISPATCHER.callback_query_handler(lambda c: c.data and c.data.lower() == "cancel", state="*")
async def cancel(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext):
    LOGGER.info("Cancelling previous command...")
    current_state = await state.get_state()
    if current_state is None:
        LOGGER.debug("was doing nothing already!")
        return
    
    LOGGER.debug(f"Cancelling state {current_state}")
    await state.finish()
    await message.reply("Cancelled.")
    await message.edit_reply_markup(reply_markup=None)
    LOGGER.debug("Cancelled.")
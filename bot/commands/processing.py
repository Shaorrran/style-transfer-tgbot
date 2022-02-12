import logging

from bot import internals

import aiogram

import skimage.io

LOGGER = logging.getLogger(__name__)

class LoadContent(aiogram.dispatcher.filters.state.StatesGroup):
    waiting_for_image = aiogram.dispatcher.filters.state.State()

class LoadStyle(aiogram.dispatcher.filters.state.StatesGroup):
    waiting_for_selection = aiogram.dispatcher.filters.state.State()
    waiting_for_image = aiogram.dispatcher.filters.state.State()
    style_names = CONFIG["styles"]["style_names"]
    style_enum = None

@internals.DISPATCHER.message_handler(commands=["/content", "/image"])
async def content(message: aiogram.types.Message) -> None:
    LOGGER.info("Processing /content command...")
    await message.reply("Upload the image you want to stylize, please\.")
    await LoadContent.waiting_for_image.set()
    LOGGER.debug("Set state to waiting for content image.")

@internals.DISPATCHER.message_handler(state=LoadContent.waiting_for_image)
async def content_upload(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext):
    LOGGER.debug("Waiting for content image...")
    if not message.photo:
        LOGGER.debug("Not a photo here, moving on.")
        await message.reply("Please upload a photo\.")
        return
    await state.update_data(content_image=skimage.io.imread(message.photo[-1].get_file().file))
    LOGGER.debug("Loaded image.")
    await message.reply(aiogram.utils.markdown.text(
        aiogram.utils.markdown.text("Content set\. Now run "),
        aiogram.utils.markdown.code("/style"),
        aiogram.utils.markdown.text(" to upload choose an image to be used as the style sample\."),
        sep=""
    ), parse_mode="MarkdownV2")
    await state.finish()
    LOGGER.debug("Finished reading content image.")

@internals.DISPATCHER.message_handler(content_types=[aiogram.types.InputMediaPhoto])
async def image_handler(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext):
    LOGGER.debug("Image posted, assuming content.")
    await message.reply("You have posted an image, so I assume this is what you want to stylize\.")
    await state.set_state(LoadContent.waiting_for_image)
    await content_upload(message, state)
    LOGGER.debug("Image handling complete, content selected.")

@internals.DISPATCHER.message_handler(commands=["/style"])
async def style(message: aiogram.types.Message) -> None:
    LOGGER.info("Processing /style command...")
    buttons = [
        aiogram.types.InlineKeyboardButton(text=f"{LoadStyle.style_names[style]}", callback_data=f"style_{style}") \
        for style in LoadStyle.styles
    ]
    LoadStyle.styles_enum = [f"style_{i}" for i, _ in enumerate(LoadStyle.style_names)]
    buttons.append(aiogram.types.InlineKeyboardButton(text="Cancel"), callback_data="cancel")
    LOGGER.debug(f"Created buttons: {buttons}")
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    keyboard.add(*buttons)
    LOGGER.debug(f"Created keyboard: {keyboard}")
    await message.reply("Please choose a given style or upload your own image\!", reply_markup=keyboard)
    await LoadStyle.waiting_for_selection.set()
    LOGGER.debug("Set state to waiting for style selection")

@internals.DISPATCHER.message_handler(state=LoadStyle.waiting_for_selection)
async def style_choose(message: aiogram.types.Message, style: aiogram.dispatcher.FSMContext):
    LOGGER.debug("Processing choice...")
    if message.text.lower() not in LoadStyle.styles_enum:
        LOGGER.debug("Incorrect choice.")
        await message.reply("Please select the style image using the keyboard\!")
        return
    await state.update_data(chosen_style=message.text.lower())
    await LoadStyle.next()
    LOGGER.debug("Set state to waiting for style image")

@internals.DISPATCHER.message_handler(state=LoadStyle.waiting_for_image)
async def style_upload(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext):
    if (await state.get_data()) == "custom":
        LOGGER.debug("Waiting for custom photo...")
        await message.reply("Upload your custom photo now\.")
        if not message.photo:
            LOGGER.debug("This is not a photo.")
            await message.reply("Please upload a photo\.")
            return
        await state.update_data(style_image=skimage.io.imread(message.photo[-1].get_file().file))
        LOGGER.debug("Processed custom style photo.")
    else:
        state_data = await state.get_data()
        LOGGER.debug(f"Current style is {state_data}")
        image = skimage.io.imread(STYLE_PATHS[state_data])
        await state.update_data(style_image=image)
        LOGGER.debug("Read corresponding image.")
    await message.reply(aiogram.utils.markdown.text(
        aiogram.utils.markdown.text("Style set\. Run "),
        aiogram.utils.markdown.code("/transfer"),
        aiogram.utils.markdown.text(" to perform the style transfer\."),
        sep=""
    ), parse_mode="MarkdownV2", reply_markup=aiogram.types.InlineKeyboardRe)
    await state.finish()
    await message.edit_reply_markup(reply_markup=None)
    LOGGER.debug("Cleaned up and finished processing state.")

@internals.DISPATCHER.message_handler(commands=["/transfer", "/run"])
async def transfer(message: aiogram.types.Message) -> None:
    LOGGER.debug("Attempting transfer...")
    LOGGER.critical("Style transfer not ready yet!")
    await message.reply("Not implemented yet.")
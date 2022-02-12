import copy
import itertools
import logging

from bot import internals

import aiogram
import aiogram.dispatcher.filters.state

import skimage.io

LOGGER = logging.getLogger(__name__)

class LoadContent(aiogram.dispatcher.filters.state.StatesGroup):
    waiting_for_image = aiogram.dispatcher.filters.state.State()

class LoadStyle(aiogram.dispatcher.filters.state.StatesGroup):
    waiting_for_selection = aiogram.dispatcher.filters.state.State()
    waiting_for_image = aiogram.dispatcher.filters.state.State()

@internals.DISPATCHER.message_handler(commands=["content", "image"])
async def content(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext) -> None:
    LOGGER.info("Processing /content command...")
    await message.reply("Upload the image you want to stylize, please\.")
    await LoadContent.waiting_for_image.set()
    LOGGER.debug("Set state to waiting for content image.")
    cur_state = await state.get_state()
    LOGGER.debug(f"current state: {cur_state}")

@internals.DISPATCHER.message_handler(state="LoadContent:waiting_for_image")
async def content_upload(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext):
    LOGGER.debug("Waiting for content image...")
    if not message.photo:
        LOGGER.debug("Not a photo here, moving on.")
        await message.reply("Please upload a photo\.")
        return
    token = internals.CONFIG["bot"]["token"]
    file_path = (await message.photo[-1].get_file()).file_path
    await state.update_data(content_image=skimage.io.imread(f"https://api.telegram.org/file/bot{token}/{file_path}"))
    LOGGER.debug("Loaded image.")
    await message.reply(aiogram.utils.markdown.text(
        aiogram.utils.markdown.text("Content set\. Now run "),
        aiogram.utils.markdown.code("/style"),
        aiogram.utils.markdown.text(" to upload choose an image to be used as the style sample\."),
        sep=""
    ), parse_mode="MarkdownV2")
    await state.finish()
    LOGGER.debug("Finished reading content image.")

@internals.DISPATCHER.message_handler(content_types=["photo"], state=None)
async def image_handler(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext):
    LOGGER.debug("Image posted, assuming content.")
    await message.reply("You have posted an image, so I assume this is what you want to stylize\.")
    await state.set_state(LoadContent.waiting_for_image)
    await content_upload(message, state)
    LOGGER.debug("Image handling complete, content selected.")

@internals.DISPATCHER.message_handler(commands=["style"])
async def style(message: aiogram.types.Message) -> None:
    LOGGER.info("Processing /style command...")
    buttons = [
        aiogram.types.InlineKeyboardButton(text=internals.STYLES[style]["display_name"], callback_data=f"style_{style}") \
        for style in internals.STYLES
    ]
    buttons.append(aiogram.types.InlineKeyboardButton(text="Cancel", callback_data="cancel"))
    LOGGER.debug(f"Created buttons: {buttons}")
    keyboard = aiogram.types.InlineKeyboardMarkup(row_width=2)
    keyboard.add(*buttons)
    images = []
    styles_inbuilt = copy.deepcopy(internals.STYLES)
    styles_inbuilt.pop("custom") # the "custom" option does not have a path to a file
    ids, data = list(styles_inbuilt.keys()), list(styles_inbuilt.values())
    ids = [ids[i:i + 8] for i in range(0, len(ids), 8)]
    data = [data[i:i + 8] for i in range(0, len(data), 8)]
    LOGGER.debug(f"ids: {ids}")
    LOGGER.debug(f"data: {data}")
    for id, setting in zip(ids, data):
        LOGGER.debug(f"checking id {id}, setting {setting}")
        images = []
        for i, e in zip(id, setting):
            LOGGER.debug(f"i: {i}, e: {e}")
            file = aiogram.types.InputFile(e["path"])
            image = aiogram.types.InputMediaPhoto(file, caption=aiogram.utils.markdown.escape_md(e["display_name"]))
            images.append(image)
        await message.reply_media_group(images)
    LOGGER.debug(f"Created keyboard: {keyboard}")
    await message.reply("Please choose a given style or upload your own image\!", reply_markup=keyboard)
    await LoadStyle.waiting_for_selection.set()
    LOGGER.debug("Set state to waiting for style selection")

@internals.DISPATCHER.message_handler(state="LoadStyle:waiting_for_selection")
async def style_choose(message: aiogram.types.Message, style: aiogram.dispatcher.FSMContext):
    LOGGER.debug("Processing choice...")
    if message.text.lower().replace("style_", "") not in internals.STYLES:
        LOGGER.debug("Incorrect choice.")
        await message.reply("Please select the style image using the keyboard\!")
        return
    await state.update_data(chosen_style=message.text.lower())
    await LoadStyle.next()
    LOGGER.debug("Set state to waiting for style image")

@internals.DISPATCHER.message_handler(state="LoadStyle:waiting_for_image")
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
        LOGGER.debug(f"Current state is {state_data}")
        image = skimage.io.imread(STYLE_PATHS[state_data])
        await state.update_data(style_image=image)
        LOGGER.debug("Read corresponding image.")
    image = await state.get_data()["style_image"]
    await message.reply_photo(aiogram.types.InputMediaPhoto(aiogram.types.BufferedInputFile(image, filename="style.jpg"), caption="Current chosen style"))
    await message.reply(aiogram.utils.markdown.text(
        aiogram.utils.markdown.text("Style set\. Run "),
        aiogram.utils.markdown.code("/transfer"),
        aiogram.utils.markdown.text(" to perform the style transfer\."),
        sep=""
    ), parse_mode="MarkdownV2")
    await state.finish()
    await message.edit_reply_markup(reply_markup=None)
    LOGGER.debug("Cleaned up and finished processing state.")

@internals.DISPATCHER.message_handler(commands=["transfer", "run"])
async def transfer(message: aiogram.types.Message) -> None:
    LOGGER.debug("Attempting transfer...")
    LOGGER.critical("Style transfer not ready yet!")
    await message.reply("Not implemented yet.")
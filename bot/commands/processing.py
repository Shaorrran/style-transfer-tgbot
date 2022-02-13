import copy
import io
import itertools
import logging

from bot import internals

import bot.style_transfer

import aiogram
import aiogram.dispatcher.filters.state

import skimage.io

LOGGER = logging.getLogger(__name__)

class LoadContent(aiogram.dispatcher.filters.state.StatesGroup):
    waiting_for_image = aiogram.dispatcher.filters.state.State()

class LoadStyle(aiogram.dispatcher.filters.state.StatesGroup):
    waiting_for_selection = aiogram.dispatcher.filters.state.State()
    waiting_for_image = aiogram.dispatcher.filters.state.State()

@dataclass
class ImageHolder:
    content = None
    style = None

@internals.DISPATCHER.message_handler(commands=["content", "image"])
async def content(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext) -> None:
    LOGGER.info("Processing /content command...")
    await message.reply("Upload the image you want to stylize, please\.")
    await LoadContent.waiting_for_image.set()
    LOGGER.debug("Set state to waiting for content image.")
    cur_state = await state.get_state()
    LOGGER.debug(f"current state: {cur_state}")

@internals.DISPATCHER.message_handler(state=LoadContent.waiting_for_image, content_types=["photo"])
@internals.DISPATCHER.callback_query_handler(state=LoadContent.waiting_for_image)
async def content_upload(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext):
    LOGGER.debug("Waiting for content image...")
    if not message.photo:
        LOGGER.debug("Not a photo here, moving on.")
        await message.reply("Please upload a photo\.")
        return
    token = internals.CONFIG["bot"]["token"]
    file_path = (await message.photo[-1].get_file()).file_path
    await state.update_data(content_image=skimage.io.imread(f"https://api.telegram.org/file/bot{token}/{file_path}"))
    ImageHolder.content = state.get_data()["content_image"]
    LOGGER.debug("Loaded image.")
    await message.reply(aiogram.utils.markdown.text(
        aiogram.utils.markdown.text("Content set\. Now run "),
        aiogram.utils.markdown.code("/style"),
        aiogram.utils.markdown.text(" to upload choose an image to be used as the style sample\."),
        sep=""
    ), parse_mode="MarkdownV2")
    await state.finish()
    LOGGER.debug("Finished reading content image.")

@internals.DISPATCHER.message_handler(content_types=["photo"])
async def image_handler(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext):
    LOGGER.debug("in image_handler")
    cur_state = await state.get_state()
    LOGGER.debug(f"Current state in image_handler: {cur_state}")
    LOGGER.debug("Image posted, assuming content.")
    await message.reply("You have posted an image, so I assume this is what you want to stylize\.")
    await state.set_state(LoadContent.waiting_for_image)
    await content_upload(message, state)
    LOGGER.debug("Image handling complete, content selected.")

@internals.DISPATCHER.message_handler(commands=["style"])
async def style(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext) -> None:
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
    cur_state = await state.get_state()
    LOGGER.debug(f"current state: {cur_state}")

@internals.DISPATCHER.callback_query_handler(lambda c: c.data.lower() != "style_custom" and c.data.lower().startswith("style_"), state=LoadStyle.waiting_for_selection)
async def style_select(query: aiogram.types.callback_query.CallbackQuery, state: aiogram.dispatcher.FSMContext):
    state_data = await state.get_data()
    LOGGER.debug(f"Current state is {state_data}")
    image = skimage.io.imread(internals.STYLES[query.data.lower().replace("style_", "")]["path"])
    await state.update_data(style_image=image)
    ImageHolder.style = image
    LOGGER.debug("Read corresponding image.")
    await query.answer("Style set\. Run /transfer or /run to perform style transfer\.")
    await query.message.edit_reply_markup(None)
    await state.finish()

@internals.DISPATCHER.callback_query_handler(lambda c: c.data.lower() == "style_custom" and c.data.lower().startswith("style_"), state=LoadStyle.waiting_for_selection)
async def style_custom(query: aiogram.types.callback_query.CallbackQuery, state: aiogram.dispatcher.FSMContext):
    await LoadStyle.next()
    await query.message.answer("Custom style selected\. Please upload an image\.")
    await query.answer("Image upload started.")
    await query.message.edit_reply_markup(None)
    cur_state = await state.get_state()
    LOGGER.debug(f"current state: {cur_state}")

@internals.DISPATCHER.message_handler(state=LoadStyle.waiting_for_image, content_types=["photo"])
@internals.DISPATCHER.callback_query_handler(state=LoadContent.waiting_for_image)
async def style_custom_upload(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext):
    LOGGER.debug("Waiting for style image...")
    if not message.photo:
        LOGGER.debug("Not a photo here, moving on.")
        await message.reply("Please upload a photo\.")
        return
    token = internals.CONFIG["bot"]["token"]
    file_path = (await message.photo[-1].get_file()).file_path
    await state.update_data(style_image=skimage.io.imread(f"https://api.telegram.org/file/bot{token}/{file_path}"))
    ImageHolder.style = state.get_data()["style_image"]
    LOGGER.debug("Loaded image.")
    await message.reply("Style set\. Run `/transfer` or `/run` to perform style transfer\.")
    await state.finish()

@internals.DISPATCHER.callback_query_handler(lambda c: not c.data.lower().startswith("style_"), state=LoadStyle.waiting_for_selection)
async def style_incorrect_choice(query: aiogram.types.callback_query.CallbackQuery, state: aiogram.dispatcher.FSMContext):
    await query.answer("You somehow chose something which should not be chosen\. Now you must live in the awful world you have created\.\.\.")

@internals.DISPATCHER.message_handler(commands=["transfer", "run"])
async def transfer(message: aiogram.types.Message) -> None:
    LOGGER.debug("Attempting transfer...")
    if not ImageHolder.content:
        await message.reply("No content image found\. Please run `/content` before attempting style transfer\.")
        return
    if not ImageHolder.style:
        await message.reply("No style image found\. Please run `/style` before attempting style transfer\.")
        return
    styled = bot.style_transfer.style_transfer_converter(ImageHolder.content, ImageHolder.style)
    if isinstance(styled, list):
        albums = [styled[i:i+8] for i in range(0, len(styled), 8)]
        for i in albums:
            media_group = []
            for j in i:
                file = aiogram.types.BufferedInputFile(j)
                image = aiogram.types.InputMediaPhoto(file, caption="Styled image")
                media_group.append(image)
            await message.answer_media_group(media_group)
    elif isinstance(styled, io.BytesIO):
        image = aiogram.types.InputMediaPhoto(styled, caption="Styled image")
        await message.reply_photo(image)
    else:
        raise ValueError("Style transfer returned something that should not have been returned.")
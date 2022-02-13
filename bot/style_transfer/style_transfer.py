import io
import logging
import typing as tp

import torch
import torchvision

import PIL

import skimage
import skimage.transform

import numpy as np

from bot import internals

IMAGE_SIZE = 512

from architectures.model import StyleTransferCNN as StyleTransferCNN

LOGGER = logging.getLogger(__name__)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

async def style_transfer(model: torch.nn.Module, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    model.eval()
    try:
        content, style = content.to(DEVICE), style.to(DEVICE)
        LOGGER.debug(f"Moved images to {DEVICE}")
        if content.dim() != 4:
            LOGGER.debug("Content turned into batch of len 1")
            content = content.unsqueeze(0)
            LOGGER.debug(f"Tensor size: {content.size()}")
        if style.dim() != 4:
            LOGGER.debug("Content turned into batch of len 1")
            style = style.unsqueeze(0)
            LOGGER.debug(f"Tensor size: {content.size()}")
        with torch.no_grad():
            styled = model.generate(content, style)
            LOGGER.debug("Forward pass done")
        styled = styled.detach().cpu()
        LOGGER.debug("Detached tensor")
    finally:
        content, style = content.cpu(), style.cpu()
        del content, style
        LOGGER.debug("Cleaned up")
    if styled.size(0) == 1: # single image
        LOGGER.debug("Batch of size 1 detected, squeezing to single image")
        styled = styled.squeeze(0)
    LOGGER.debug(f"Styled tensor size: {styled.size()}")
    return styled

async def style_transfer_converter(content: tp.Union[np.ndarray, PIL.Image.Image], style: tp.Union[np.ndarray, PIL.Image.Image]) -> tp.Union[tp.List[io.BytesIO], io.BytesIO]:
    LOGGER.debug("in converter")
    model = torch.load(internals.CONFIG["model"]["model_path"], map_location=DEVICE)
    LOGGER.debug("model loaded")
    content = skimage.transform.resize(content, (IMAGE_SIZE, IMAGE_SIZE))
    content = torchvision.transforms.ToTensor()(content).to(torch.float)
    LOGGER.debug(f"Content is now of type {type(content)}")
    style = skimage.transform.resize(style, (IMAGE_SIZE, IMAGE_SIZE))
    style = torchvision.transforms.ToTensor()(style).to(torch.float)
    LOGGER.debug(f"Content is now of type {type(style)}")
    styled = await style_transfer(model, content, style)
    LOGGER.debug(f"Output tensor dim: {styled.dim()}")
    if styled.dim() == 3:
        LOGGER.debug("Got single image")
        styled = PIL.Image.fromarray(skimage.img_as_ubyte(np.clip(styled.permute(1, 2, 0).cpu().numpy(), 0, 1)))
        styled_bytes = io.BytesIO()
        styled.save(styled_bytes, format="JPEG")
        LOGGER.debug("Turned into BytesIO")
        styled_bytes.seek(0)
        return styled_bytes
    else:
        LOGGER.debug("Got batch of images")
        styled = list(styled)
        bytesio_arr = []
        for i in styled:
            image =  PIL.Image.fromarray(skimage.img_as_ubyte(np.clip(i.permute(1, 2, 0).cpu().numpy(), 0, 1)))
            styled_bytes = io.BytesIO()
            styled.save(styled_bytes, format="JPEG")
            styled_bytes.seek(0)
            bytesio_arr.append(styled_bytes)
        return bytesio_arr
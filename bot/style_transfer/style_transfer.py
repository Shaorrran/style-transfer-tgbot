import io
import typing as tp

import torch
import torchvision

import PIL

import numpy as np

from bot import internals

from architectures.model import StyleTransferCNN


DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def style_transfer(model: torch.nn.Module, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    model.eval()
    try:
        content, style = content.to(DEVICE), style.to(DEVICE)
        if content.dim() != 4:
            content = content.unsqueeze(0)
        if style.dim() != 4:
            style = style.unsqueeze(0)
        with torch.no_grad():
            styled = model.generate(content, style)
        styled = styled.detach().cpu()
    finally:
        content, style = content.cpu(), style.cpu()
        del content, style
    if styled.size(0) == 1: # single image
        styled = styled.squeeze(0)
    return styled

def style_transfer_converter(content: tp.Union[np.ndarray, PIL.Image], style: tp.Union[np.ndarray, PIL.Image]) -> tp.Union[tp.List[io.BytesIO], io.BytesIO]:
    model = torch.load(internals.CONFIG["model"]["model_path"])
    content = torchvision.transforms.ToTensor()(content)
    style = torchvision.transforms.ToTensor()(style)
    styled = style_transfer(model, content, style)
    if styled.dim == 3:
        styled = PIL.Image(styled.permute(1, 2, 0).cpu().numpy())
        styled_bytes = io.BytesIO()
        styled.save(styled_bytes, format="JPG")
        return styled_bytes
    else:
        styled = list(styled)
        bytesio_arr = []
        for i in styled:
            image = PIL.Image(i.permute(1, 2, 0).cpu().numpy())
            styled_bytes = io.BytesIO()
            styled.save(styled_bytes, format="JPG")
            bytesio_arr.append(styled_bytes)
        return bytesio_arr
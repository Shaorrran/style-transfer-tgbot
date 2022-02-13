import collections

import torch

class Decoder(torch.nn.Module):
    def __init__(self, level, pretrained_path=None):
        super().__init__()
        decoder_layers = collections.OrderedDict([
            ("reflectpad0", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv0", torch.nn.Conv2d(512, 512, kernel_size=3)),
            ("act0", torch.nn.ReLU(inplace=True)),
            ("upsample0", torch.nn.Upsample(scale_factor=2)),
            ("reflectpad1", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv1", torch.nn.Conv2d(512, 512, kernel_size=3)),
            ("act1", torch.nn.ReLU(inplace=True)),
            ("reflectpad2", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv2", torch.nn.Conv2d(512, 512, kernel_size=3)),
            ("act2", torch.nn.ReLU(inplace=True)),
            ("reflectpad3", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv3", torch.nn.Conv2d(512, 512, kernel_size=3)),
            ("act3", torch.nn.ReLU(inplace=True)),
            ("reflectpad4", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv4", torch.nn.Conv2d(512, 256, kernel_size=3)),
            ("act4", torch.nn.ReLU(inplace=True)),
            ("upsample1", torch.nn.Upsample(scale_factor=2)),
            ("reflectpad5", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv5", torch.nn.Conv2d(256, 256, kernel_size=3)),
            ("act5", torch.nn.ReLU(inplace=True)),
            ("reflectpad6", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv6", torch.nn.Conv2d(256, 256, kernel_size=3)),
            ("act6", torch.nn.ReLU(inplace=True)),
            ("reflectpad7", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv7", torch.nn.Conv2d(256, 256, kernel_size=3)),
            ("act7", torch.nn.ReLU(inplace=True)),
            ("reflectpad8", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv8", torch.nn.Conv2d(256, 128, kernel_size=3)),
            ("act8", torch.nn.ReLU(inplace=True)),
            ("upsample2", torch.nn.Upsample(scale_factor=2)),
            ("reflectpad9", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv9", torch.nn.Conv2d(128, 128, kernel_size=3)),
            ("act9", torch.nn.ReLU(inplace=True)),
            ("reflectpad10", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv10", torch.nn.Conv2d(128, 64, kernel_size=3)),
            ("act10", torch.nn.ReLU(inplace=True)),
            ("upsample3", torch.nn.Upsample(scale_factor=2)),
            ("reflectpad11", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv11", torch.nn.Conv2d(64, 64, kernel_size=3)),
            ("act11", torch.nn.ReLU(inplace=True)),
            ("reflectpad12", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv12", torch.nn.Conv2d(64, 3, kernel_size=3))
        ])
        if level == 1:
            self.layers = torch.nn.Sequential(collections.OrderedDict(list(decoder_layers.items())[-2:]))
        elif level == 2:
            self.layers = torch.nn.Sequential(collections.OrderedDict(list(decoder_layers.items())[-9:]))
        elif level == 3:
            self.layers = torch.nn.Sequential(collections.OrderedDict(list(decoder_layers.items())[-16:]))
        elif level == 4:
            self.layers = torch.nn.Sequential(collections.OrderedDict(list(decoder_layers.items())[-29:]))
        elif level == 5:
            self.layer = torch.nn.Sequential(decoder_layers)
        else:
            raise ValueError("level should be between 1 and 5")
        if pretrained_path is not None:
            self._load_weights(torch.load(pretrained_path, map_location=lambda storage, loc: storage))

    def _load_weights(self, state_dict):
        self_dict = self.state_dict()
        self_dict = {k: v for k, v in zip(self_dict.keys(), state_dict.values())}
        self.load_state_dict(self_dict)

    def forward(self, x):
        return self.layers(x)
import collections

import torch

class NormalizedVGG(torch.nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.tail = torch.nn.Sequential(collections.OrderedDict([
            ("conv_initial", torch.nn.Conv2d(3, 3, kernel_size=1)),
            ("reflectpad_tail", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_tail", torch.nn.Conv2d(3, 64, kernel_size=3)),
            ("act_tail", torch.nn.ReLU(inplace=True))
        ]))
        self.lower_spine = torch.nn.Sequential(collections.OrderedDict([
            ("reflectpad_lspine0", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_lspine0", torch.nn.Conv2d(64, 64, kernel_size=3)),
            ("act_lspine0", torch.nn.ReLU(inplace=True)),
            ("maxpool_lspine", torch.nn.MaxPool2d(2, ceil_mode=True)),
            ("reflectpad_lspine1", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_lspine1", torch.nn.Conv2d(64, 128, kernel_size=3)),
            ("act_lspine1", torch.nn.ReLU(inplace=True))
        ]))
        self.spine = torch.nn.Sequential(collections.OrderedDict([
            ("reflectpad_spine0", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_spine0", torch.nn.Conv2d(128, 128, kernel_size=3)),
            ("act_spine0", torch.nn.ReLU(inplace=True)),
            ("maxpool_spine", torch.nn.MaxPool2d(2, ceil_mode=True)),
            ("reflectpad_spine1", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_spine1", torch.nn.Conv2d(128, 256, kernel_size=3)),
            ("act_spine1", torch.nn.ReLU(inplace=True))
        ]))
        self.neck = torch.nn.Sequential(collections.OrderedDict([
            ("reflectpad_neck0", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_neck0", torch.nn.Conv2d(256, 256, kernel_size=3)),
            ("act_neck0", torch.nn.ReLU(inplace=True)),
            ("reflectpad_neck1", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_neck1", torch.nn.Conv2d(256, 256, kernel_size=3)),
            ("act_neck1", torch.nn.ReLU(inplace=True)),
            ("reflectpad_neck2", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_neck2", torch.nn.Conv2d(256, 256, kernel_size=3)),
            ("act_neck2", torch.nn.ReLU(inplace=True)),
            ("maxpool_neck", torch.nn.MaxPool2d(2, ceil_mode=True)),
            ("reflectpad_neck3", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_neck3", torch.nn.Conv2d(256, 512, kernel_size=3)),
            ("act_neck3", torch.nn.ReLU(inplace=True))
        ]))
        self.head = torch.nn.Sequential(collections.OrderedDict([    
            ("reflectpad_head0", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_head0", torch.nn.Conv2d(512, 512, kernel_size=3)),
            ("act_head0", torch.nn.ReLU(inplace=True)),
            ("reflectpad_head1", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_head1", torch.nn.Conv2d(512, 512, kernel_size=3)),
            ("act_head1", torch.nn.ReLU(inplace=True)),
            ("reflectpad_head2", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_head2", torch.nn.Conv2d(512, 512, kernel_size=3)),
            ("act_head2", torch.nn.ReLU(inplace=True)),
            ("maxpool_head", torch.nn.MaxPool2d(2, ceil_mode=True)),
            ("reflectpad_head3", torch.nn.ReflectionPad2d((1, 1, 1, 1))),
            ("conv_head3", torch.nn.Conv2d(512, 512, kernel_size=3)),
            ("act_head3", torch.nn.ReLU(inplace=True))
        ]))
        if pretrained_path is not None:
            self._load_weights(torch.load(pretrained_path, map_location=lambda storage, loc: storage))

        for p in self.parameters():
            p.requires_grad = False
    
    def _load_weights(self, state_dict):
        self_dict = self.state_dict()
        self_dict = {k: v for k, v in zip(self_dict.keys(), state_dict.values())}
        self.load_state_dict(self_dict)
        
    def forward(self, x, target="head", output_last_feature=True):
        x_tail = self.tail(x)
        x_lower_spine = self.lower_spine(x_tail)
        x_spine = self.spine(x_lower_spine)
        x_neck = self.neck(x_spine)
        x_head = self.head(x_neck)
        if output_last_feature:
            if target == "tail":
                return x_tail
            if target == "lower_spine":
                return x_lower_spine
            if target == "spine":
                return x_spine
            if target == "neck":
                return x_neck
            return x_head
        else:
            if target == "tail":
                return x_tail
            if target == "lower_spine":
                return x_tail, x_lower_spine
            if target == "spine":
                return x_tail, x_lower_spine, x_spine
            if target == "neck":
                return x_tail, x_lower_spine, x_spine, x_neck
            return x_tail, x_lower_spine, x_spine, x_neck, x_head
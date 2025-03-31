
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
import torchvision
import math


class AIM_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.5, act_layer=nn.ReLU, skip_connect=True, type='conv', kernel=3, g=1):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.type=type
        kernel=kernel
        if self.type=='conv':
            self.D_fc1 = nn.Conv2d(D_features, D_hidden_features, kernel_size=(kernel,kernel), padding=kernel//2, groups=g)
            self.D_mapping = nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=(kernel,kernel), padding=kernel//2)
            self.D_fc2 = nn.Conv2d(D_hidden_features, D_features, kernel_size=(kernel,kernel), padding=kernel//2, groups=g)
        else:
            self.D_fc1 = nn.Linear(D_features, D_hidden_features)
            self.D_mapping = nn.Linear(D_hidden_features, D_hidden_features)
            self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
        nn.init.constant_(self.D_mapping.weight, 0.)
        nn.init.constant_(self.D_mapping.bias, 0.)
        nn.init.constant_(self.D_fc1.bias, 0.)
        nn.init.constant_(self.D_fc2.bias, 0.)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_mapping(xs)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x



### modified from https://github.com/pytorch/vision/blob/423a1b0ebdea077cc69478812890845741048d2e/torchvision/models/resnet.py#L334

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CustomResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        depth: int = 50,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.kwargs=kwargs

        self.depth = depth

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.ds_rate=kwargs['ds_rate']

        self.groups = groups
        self.base_width = width_per_group
        if self.ds_rate==1.0:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.ds_rate in [0.5, 1.0]:
            # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.maxpool = nn.Identity()
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        if self.ds_rate in [0.25, 0.5, 1.0]:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], ds_rate=self.ds_rate)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], ds_rate=self.ds_rate)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], ds_rate=self.ds_rate)
        else:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        if self.ds_rate in [0.25, 0.5, 1.0]:
            self.avgpool = nn.Identity()
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        

        Adapter=AIM_Adapter

        if self.depth == 50:
            C1, C2, C3 = 64, 512, 2048

        if 'early' in kwargs['adapter']:
            self.early_adapter_1 = Adapter(C1)
            self.early_adapter_2 = Adapter(C1)
            self.early_adapter_3 = Adapter(C1)
        if 'middle' in kwargs['adapter']:
            self.middle_adapter_1 = Adapter(C2)
            self.middle_adapter_2 = Adapter(C2)
            self.middle_adapter_3 = Adapter(C2)
        if 'avg.late' in kwargs['adapter']:
            #self.late_adapter = AIM_Adapter(2048)
            self.late_adapter_1 = Adapter(C3, mlp_ratio=0.25, type='linear')
            self.late_adapter_2 = Adapter(C3, mlp_ratio=0.25, type='linear')
            self.late_adapter_3 = Adapter(C3, mlp_ratio=0.25, type='linear')
        elif 'late' in kwargs['adapter']:
            #self.late_adapter = AIM_Adapter(2048)
            self.late_adapter_1 = Adapter(C3, mlp_ratio=0.25, type='conv')
            self.late_adapter_2 = Adapter(C3, mlp_ratio=0.25, type='conv')
            self.late_adapter_3 = Adapter(C3, mlp_ratio=0.25, type='conv')
        elif 'late.layer.3.k.1.down.4.g.8' in kwargs['adapter']:
            self.late_adapter_1 = Adapter(C3, mlp_ratio=0.25, type='conv', kernel=1, g=8)
            self.late_adapter_2 = Adapter(C3, mlp_ratio=0.25, type='conv', kernel=1, g=8)
            self.late_adapter_3 = Adapter(C3, mlp_ratio=0.25, type='conv', kernel=1, g=8)
        elif 'late.layer.3.k.1.down.2.g.1' in kwargs['adapter']:
            self.late_adapter_1 = Adapter(C3, mlp_ratio=0.5, type='conv', kernel=1, g=1)
            self.late_adapter_2 = Adapter(C3, mlp_ratio=0.5, type='conv', kernel=1, g=1)
            self.late_adapter_3 = Adapter(C3, mlp_ratio=0.5, type='conv', kernel=1, g=1)
            

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        ds_rate = 0.0,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if ds_rate in [0.25, 0.5, 1.0]:
                stride = 1
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if 'early' in self.kwargs['adapter']:
            x=self.early_adapter_1(x)
            x=self.early_adapter_2(x)
            x=self.early_adapter_3(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if 'middle' in self.kwargs['adapter']:
            x=self.middle_adapter_1(x)
            x=self.middle_adapter_2(x)
            x=self.middle_adapter_3(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        if 'late' in self.kwargs['adapter'] or 'late.layer.3.k.1.down.4.g.8' in self.kwargs['adapter'] or 'late.layer.3.k.1.down.2.g.1' in self.kwargs['adapter']:
            x=self.late_adapter_1(x)
            x=self.late_adapter_2(x)
            x=self.late_adapter_3(x)
        x = self.avgpool(x)
        if self.ds_rate == 0:
            x = torch.flatten(x, 1)
        # if 'avg.late' in self.kwargs['adapter']:
        #     x=self.late_adapter_1(x)
        #     x=self.late_adapter_2(x)
        #     x=self.late_adapter_3(x)
        x = self.fc(x)

        return x



    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights = None,
    progress = None,
    depth = 50,
    norm_layer = None,
    **kwargs: Any,
):
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = CustomResNet(block, layers, depth, norm_layer=norm_layer,  **kwargs)


    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model



def resnet50(pretrained=None, **kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)



if __name__=="__main__":
    pass
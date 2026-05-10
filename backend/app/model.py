from torch import nn
from torchvision import models


class SimpleCifarNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


SUPPORTED_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "densenet121",
    "convnext_tiny",
    "vit_b_16",
    "squeezenet1_0",
    "simple_cnn",
]


def get_supported_models() -> list[str]:
    return SUPPORTED_MODELS.copy()


def _replace_classifier(model: nn.Module, model_name: str, num_classes: int) -> None:
    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return

    if model_name.startswith("mobilenet_v3") or model_name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return

    if model_name == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return

    if model_name == "convnext_tiny":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return

    if model_name == "vit_b_16":
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return

    if model_name == "squeezenet1_0":
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
        return

    if model_name == "simple_cnn":
        model.classifier[-1] = nn.Linear(128, num_classes)
        return

    raise ValueError(f"Unsupported model: {model_name}")


def create_model(
    model_name: str = "resnet18",
    num_classes: int = 10,
    transfer_learning: bool = True,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "simple_cnn":
        model = SimpleCifarNet()
    else:
        weights = None
        if transfer_learning:
            try:
                if model_name == "resnet18":
                    weights = models.ResNet18_Weights.IMAGENET1K_V1
                elif model_name == "resnet34":
                    weights = models.ResNet34_Weights.IMAGENET1K_V1
                elif model_name == "resnet50":
                    weights = models.ResNet50_Weights.IMAGENET1K_V2
                elif model_name == "mobilenet_v3_small":
                    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                elif model_name == "mobilenet_v3_large":
                    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
                elif model_name == "efficientnet_b0":
                    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
                elif model_name == "densenet121":
                    weights = models.DenseNet121_Weights.IMAGENET1K_V1
                elif model_name == "convnext_tiny":
                    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
                elif model_name == "vit_b_16":
                    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
                elif model_name == "squeezenet1_0":
                    weights = models.SqueezeNet1_0_Weights.IMAGENET1K_V1
            except Exception:
                weights = None

        try:
            if model_name == "resnet18":
                model = models.resnet18(weights=weights)
            elif model_name == "resnet34":
                model = models.resnet34(weights=weights)
            elif model_name == "resnet50":
                model = models.resnet50(weights=weights)
            elif model_name == "mobilenet_v3_small":
                model = models.mobilenet_v3_small(weights=weights)
            elif model_name == "mobilenet_v3_large":
                model = models.mobilenet_v3_large(weights=weights)
            elif model_name == "efficientnet_b0":
                model = models.efficientnet_b0(weights=weights)
            elif model_name == "densenet121":
                model = models.densenet121(weights=weights)
            elif model_name == "convnext_tiny":
                model = models.convnext_tiny(weights=weights)
            elif model_name == "vit_b_16":
                model = models.vit_b_16(weights=weights)
            elif model_name == "squeezenet1_0":
                model = models.squeezenet1_0(weights=weights)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception:
            if model_name == "resnet18":
                model = models.resnet18(weights=None)
            elif model_name == "resnet34":
                model = models.resnet34(weights=None)
            elif model_name == "resnet50":
                model = models.resnet50(weights=None)
            elif model_name == "mobilenet_v3_small":
                model = models.mobilenet_v3_small(weights=None)
            elif model_name == "mobilenet_v3_large":
                model = models.mobilenet_v3_large(weights=None)
            elif model_name == "efficientnet_b0":
                model = models.efficientnet_b0(weights=None)
            elif model_name == "densenet121":
                model = models.densenet121(weights=None)
            elif model_name == "convnext_tiny":
                model = models.convnext_tiny(weights=None)
            elif model_name == "vit_b_16":
                model = models.vit_b_16(weights=None)
            elif model_name == "squeezenet1_0":
                model = models.squeezenet1_0(weights=None)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

    _replace_classifier(model, model_name, num_classes)

    return model

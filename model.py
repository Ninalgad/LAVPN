from model_utils import *
import torchvision


class SegmentationModel(nn.Module):
    def __init__(self, output_dim=1, hidden_dim=64, pretrained=True):
        super().__init__()
        self.encoder = UNet16(num_classes=hidden_dim, pretrained=pretrained)
        self.hidden_layer = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.output_layer = nn.Conv2d(hidden_dim, output_dim, 1)

    def set_trainable_encoder(self, mode=True):
        for param in self.encoder.encoder.parameters():
            param.requires_grad = mode

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.ReLU()(x)
        x = self.hidden_layer(x)
        x = torch.nn.ReLU()(x)
        return self.output_layer(x)


class ClassifierModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ClassifierModel, self).__init__()
        weights = None
        if pretrained:
            weights = 'IMAGENET1K_V1'
        self.encoder = torchvision.models.vgg16(weights=weights).features
        self.head = point_wise_feed_forward_network(512, 1, 128)

    def forward(self, x):
        x = self.encoder(x)

        # global max pool
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.squeeze(x, dim=(2, 3))

        return self.head(x)

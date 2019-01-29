import torch

# config
data_set = 'mnist'
n_epochs = 40
n_feature = 2
weight_decay = 1e-3
scale = 20


class Model(torch.nn.Module):

    def __init__(self, feature):
        super(Model, self).__init__()
        self.backbone = torch.nn.Sequential(

            # 28 - 14
            torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),

            # 14 - 7
            torch.nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),

            # 7 - 3
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            # 3 - 1
            torch.nn.AvgPool2d(stride=4, kernel_size=4),

        )

        self.feature = torch.nn.Linear(512, feature)  # Feature extract layer
        self.pred = torch.nn.Linear(feature, 10, bias=False)  # Classification layer

        # for m in self.modules():
        #    if isinstance(m, torch.nn.Conv2d):
        #        torch.nn.init.xavier_normal_(m.weight)
        #    elif isinstance(m, torch.nn.BatchNorm2d):
        #        torch.nn.init.constant_(m.weight, 1)
        #        torch.nn.init.constant_(m.bias, 0)


    def forward(self, x):

        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.feature(x)

        feature_norm = torch.norm(x, dim=1, keepdim=True)
        x = scale * x / feature_norm
        x = self.pred(x)

        return x

class FeatureExtractor(torch.nn.Module):

    def __init__(self, submodule, extracted_layers):
            super(FeatureExtractor, self).__init__()
            self.submodule = submodule
            self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name,module in self.submodule._modules.items():
            if name is "feature": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs[name] = x
        return outputs
import torch
import torch.nn as nn
import torchvision.models as models
from .flame import FLAME
from ..utils.utils import orthographic_projection, transform_landmarks


class Encoder(nn.Module):
    """
        Network class combining FLAME and ResNet layers
    """

    def __init__(self, args, device):
        super(Encoder, self).__init__()

        # Initialize the Resnet50 Encoder
        # 159 = 3 global rotation(pose) + 3 jaw rotation(pose) + 1 scale (cam) + 2 rotation (cam) + 50 (expression) + 100 (shape)
        self.resnet = ResnetEncoder(pretrained=False, num_classes=args.num_classes).to(device)

        # Initialize the flame layer
        self.flame_layer = FLAME(args).to(device)
        # Set FLAME on evaluation mode
        self.flame_layer.eval()

    def forward(self, input_img, tf_params):
        # Get encoded features in latent space
        encoded_features = self.resnet(input_img)

        self.pose_params, self.scale, self.rot, self.expression_params, self.shape_params = encoded_features[:,0:6], encoded_features[:,6], encoded_features[:,7:9], encoded_features[:,9:59], encoded_features[:,59:159]

        # Acquire vertices and landmarks from the flame layer with encoded features
        vertices, landmarks, _ = self.flame_layer(self.shape_params, self.expression_params, self.pose_params)

        # Transform FLAME landmarks same ratio as original image landmarks
        transformed_landmarks = transform_landmarks(landmarks, tf_params)

        # Project landmarks on 2D with scaling and translation parameters
        projected_landmarks = orthographic_projection(transformed_landmarks, self.scale, self.rot)

        return projected_landmarks, vertices


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
       Modified from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, num_classes, pretrained=False):
        super(ResnetEncoder, self).__init__()
        self.num_parameters = num_classes
        self.encoder = models.resnet50(pretrained)

        # Fully connected layer
        # After flattening, the dimension is [1, 2048]
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, input_image):
        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

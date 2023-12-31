import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoder_layer_sizes, activation):
        super(Autoencoder, self).__init__()
        self.encoder = self._generate_layers(input_dim, encoder_layer_sizes, activation)
        self.decoder = self._generate_layers(encoder_layer_sizes[-1], encoder_layer_sizes, activation, reverse=True)

    def _generate_layers(self, input_dim, layer_sizes, activation, reverse=False):
        layers = []
        if reverse:
            layer_sizes = layer_sizes[0:-1][::-1]
            layer_sizes.append(self.encoder[0].in_features)
        for size in layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(activation)
            input_dim = size
        layers = layers[0:-1]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
class Classifier(nn.Module):
    def __init__(self, encoder):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(self.encoder[-1].out_features, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.sigmoid(self.fc(x))
        return x

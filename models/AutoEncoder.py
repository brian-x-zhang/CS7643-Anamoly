import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoder_layer_sizes, decoder_layer_sizes):
        super(Autoencoder, self).__init__()
        self.encoder = self._generate_layers(input_dim, encoder_layer_sizes, nn.ReLU())
        self.decoder = self._generate_layers(encoder_layer_sizes[-1], decoder_layer_sizes, nn.ReLU(), reverse=True)

    def _generate_layers(self, input_dim, layer_sizes, activation, reverse=False):
        layers = []
        if reverse:
            layer_sizes = layer_sizes[::-1]  
        for size in layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(activation())
            input_dim = size
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
# from networks.BadEncoderOriginalModels.simclr_model import SimCLR, SimCLRBase
# from networks.BadEncoderOriginalModels.nn_classifier import NeuralNet
import torch


class BadEncoderFullModelAdaptivePartialModel(torch.nn.Module):

    def __init__(self, encoder, classifier, inspect_layer_position=1,
                 original_input_img_shape=(1, 3, 32, 32)):
        super(BadEncoderFullModelAdaptivePartialModel, self).__init__()
        self.encoder = encoder
        self.segment_encoder_layer = int(len(self.encoder.f.f)*0.5) # len(self.encoder.f.f)*0.5
        print('Segment_encoder_layer: ', self.segment_encoder_layer)
        print('All_encoder_layer: ', int(len(self.encoder.f.f)))
        self.classifier = classifier
        # inspect_layer_position indicates which layer to inspect on.
        self.inspect_layer_positions = [0, 1, 2]
        self.inspect_layer_position = self.inspect_layer_positions[inspect_layer_position]
        self.input_shapes = []
        template_original_input = torch.ones(size=original_input_img_shape)
        self._forward_record_input_shapes(template_original_input)

        self.use_adaptive_forward = True

    def forward(self, x):
        if self.use_adaptive_forward:
            return self.adaptive_forward(x)
        else:
            return super().forward(x)

    def adaptive_forward(self, x):
        if self.inspect_layer_position in [0]:
            for layer_i in range(len(self.encoder.f.f)):
                if layer_i < self.segment_encoder_layer:
                    x = self.encoder.f.f[layer_i](x)
        if self.inspect_layer_position in [0, 1]:
            for layer_i in range(len(self.encoder.f.f)):
                if layer_i >= self.segment_encoder_layer:
                    x = self.encoder.f.f[layer_i](x)
        # if self.inspect_layer_position in [0, 1]:
        #     x = self.encoder.g(x)
        if self.inspect_layer_position in [0, 1, 2]:
            x = torch.flatten(x, start_dim=1)
            x = self.classifier(x)
        return x

    def _forward_record_input_shapes(self, x):
        self.eval()
        self.input_shapes.append(x.shape)
        for layer_i in range(len(self.encoder.f.f)):
            if layer_i < self.segment_encoder_layer:
                x = self.encoder.f.f[layer_i](x)
        self.input_shapes.append(x.shape)
        for layer_i in range(len(self.encoder.f.f)):
            if layer_i >= self.segment_encoder_layer:
                x = self.encoder.f.f[layer_i](x)
        self.input_shapes.append(x.shape)
        # x = self.encoder.g(x)
        # self.input_shapes.append(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        self.train()
        return x

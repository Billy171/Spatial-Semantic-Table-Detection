from layoutlmft.models.layoutlmv2 import LayoutLMv2Model
from model.decoder import TableDetectionDecoder
import torch
from torch import nn
from detectron2.structures import ImageList


class TableDetectionModel(nn.Module):
    """
    This module performs table detection given preprocessed inputs of documents
    """
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")
        self.encoder.to(self.device)
        self.decoder = TableDetectionDecoder(config)
        

    def forward(self, examples):
        input_ids = examples['token_ids'].to(self.device)
        bbox = examples['bboxes'].to(self.device)
        image = ImageList(examples['image'].to(self.device), image_sizes=[(224, 224)])
        h = self.encoder(input_ids=input_ids,
                         bbox=bbox,
                         image=image,
                         output_hidden_states=True,
                         )
        outputs = self.decoder(h.last_hidden_state)
        return outputs

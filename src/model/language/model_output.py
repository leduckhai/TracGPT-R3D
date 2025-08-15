from dataclasses import dataclass
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Dict
import torch
@dataclass
class TracVisionModelOutput(BaseModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    aux_loss: Optional[Dict[str, torch.FloatTensor]] = None
    predicts:Optional[torch.FloatTensor] = None
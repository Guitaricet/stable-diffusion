import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


class FrozenHugEmbedderWithAdapter(nn.Module):
    def __init__(self, model_name, max_length, output_dim, **model_kwargs):
        super().__init__()
        # how to load just the encoder of t5?
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.freeze()  # freeze the transformer

        if hasattr(self.transformer.config, "d_model"):
            transformer_hidden = self.transformer.config.d_model
        elif hasattr(self.transformer.config, "word_embed_proj_dim"):
            transformer_hidden = self.transformer.config.word_embed_proj_dim
        elif hasattr(self.transformer.config, "hidden_size"):
            transformer_hidden = self.transformer.config.hidden_size
        else:
            raise ValueError("Could not find transformer hidden size")

        self.blank_conditioning = nn.Embedding(max_length, output_dim)
        self.adapter = nn.Sequential(
            nn.Linear(transformer_hidden, 4 * transformer_hidden),
            nn.SiLU(),  # SiLU(x) = Swish(x) = x * sigmoid(x)
            nn.Linear(4 * transformer_hidden, output_dim),
        ) # only the adapter is trainable

        self.model_name = model_name
        self.max_length = max_length
        self.output_dim = output_dim

    @property
    def device(self):
        return self.transformer.device

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def get_blank_conditioning(self, batch_size, seq_len):
        return self.blank_conditioning.weight[:seq_len].repeat(batch_size, 1, 1)

    def forward(self, text, pad_to_length=None):
        kwargs = {"padding": "max_length"} if pad_to_length is not None else {}

        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=pad_to_length or self.max_length,
            padding=True,
            return_tensors="pt",
            **kwargs,
        )
        batch_encoding = batch_encoding.to(self.device)
        outputs = self.transformer(**batch_encoding)

        z = outputs.last_hidden_state
        return self.adapter(z)

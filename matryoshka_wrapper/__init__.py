import torch
from torch import nn
from peft import PeftModel

# ===== Matryoshka Wrapper =====
class MatryoshkaWrapper(nn.Module):
    def __init__(self, peft_model, dims=[8, 64, 128, 256]):
        super().__init__()
        self.peft_model = peft_model
        self.base_dim = peft_model.config.hidden_size

        self.projections = nn.ModuleDict({
            str(dim): self._create_projection(dim) for dim in dims
        })

        self._device = next(self.parameters()).device if any(p is not None for p in self.parameters()) \
                      else torch.device('cpu')

    @property
    def device(self):
        return self._device

    def _create_projection(self, dim):
        if dim == self.base_dim:
            return nn.Identity()
        elif dim >= 128:
            return nn.Linear(self.base_dim, dim)
        else:
            return nn.Sequential(
                nn.Linear(self.base_dim, 128),
                nn.ReLU(),
                nn.Linear(128, dim)
            )

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.peft_model(input_ids, attention_mask)
        base_emb = outputs.last_hidden_state.mean(dim=1)

        return {
            str(dim): proj(base_emb) for dim, proj in self.projections.items()
        }

    def to(self, device):
        super().to(device)
        self._device = device
        return self


def get_embeddings(text, model, tokenizer, dim="256"):
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False
    ).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'])
        emb = outputs[dim]
        if emb.dim() > 2:
            emb = emb.squeeze(0)

    return emb

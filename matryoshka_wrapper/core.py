import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from torch import nn
import os
from huggingface_hub import hf_hub_download

class MatryoshkaWrapper(nn.Module):
    """
    Wrapper for creating Matryoshka embeddings with multiple dimensions
    from a PEFT model.
    """
    
    def __init__(self, peft_model, dims=[8, 64, 128, 256]):
        super().__init__()
        self.peft_model = peft_model
        self.base_dim = (peft_model.config.hidden_size 
                        if hasattr(peft_model.config, 'hidden_size') 
                        else peft_model.base_model.config.hidden_size)

        self.projections = nn.ModuleDict({
            str(dim): self._create_projection(dim) for dim in dims
        })
        self._device = (next(self.parameters()).device 
                       if any(p is not None for p in self.parameters()) 
                       else torch.device('cpu'))

    @property
    def device(self):
        return self._device

    def _create_projection(self, dim):
        """Create projection layer for specific dimension"""
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
        """Forward pass through the model"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.peft_model(input_ids, attention_mask)
        base_emb = outputs.last_hidden_state.mean(dim=1)
        return {str(dim): proj(base_emb) for dim, proj in self.projections.items()}

    def to(self, device):
        """Move model to device"""
        super().to(device)
        self._device = device
        return self

    def get_embedding(self, text, tokenizer, dim="256"):
        """
        Get embedding for text with specified dimension
        
        Args:
            text (str): Input text
            tokenizer: Tokenizer instance
            dim (str): Embedding dimension
            
        Returns:
            torch.Tensor: Embedding vector
        """
        self.eval()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            return_token_type_ids=False
        ).to(self.device)

        with torch.no_grad():
            outputs = self(input_ids=inputs['input_ids'], 
                          attention_mask=inputs['attention_mask'])
            emb = outputs[str(dim)]
            if emb.dim() > 2:
                emb = emb.squeeze(0)
        return emb


def load_model(repo_id, dim="256", base_model_name="aubmindlab/bert-base-arabertv02"):
    """
    Load Matryoshka model from Hugging Face repository
    
    Args:
        repo_id (str): Repository ID on Hugging Face
        dim (str): Default embedding dimension
        base_model_name (str): Base model name
        
    Returns:
        tuple: (model, tokenizer)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    base_model = AutoModel.from_pretrained(base_model_name).to(device)
    peft_model = PeftModel.from_pretrained(base_model, repo_id).to(device)
    model = MatryoshkaWrapper(peft_model).to(device)

    try:
        wrapper_weights_path = hf_hub_download(repo_id=repo_id, 
                                             filename="matryoshka_wrapper.pt")
        state_dict = torch.load(wrapper_weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded matryoshka_wrapper.pt from {repo_id}")
    except Exception as e:
        print(f"Could not load matryoshka_wrapper.pt from Hugging Face Hub: {e}")

    model.eval()
    return model, tokenizer

import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from torch import nn
import os # ستحتاج إلى استيراد os للتعامل مع المسارات

class MatryoshkaWrapper(nn.Module):
    def __init__(self, peft_model, dims=[8, 64, 128, 256]):
        super().__init__()
        self.peft_model = peft_model
        # تأكد أن peft_model لديه config.hidden_size
        self.base_dim = peft_model.config.hidden_size if hasattr(peft_model.config, 'hidden_size') else peft_model.base_model.config.hidden_size

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
        # peft_model هو الآن النموذج الأساسي مع الـ adapter
        outputs = self.peft_model(input_ids, attention_mask)
        # إذا كان النموذج عبارة عن PEFTModel تم دمجها، فإن outputs.last_hidden_state هو ما نريده
        base_emb = outputs.last_hidden_state.mean(dim=1)
        return {str(dim): proj(base_emb) for dim, proj in self.projections.items()}

    def to(self, device):
        super().to(device)
        self._device = device
        return self

    def get_embedding(self, text, tokenizer, dim="256"):
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
            outputs = self(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            emb = outputs[str(dim)]
            if emb.dim() > 2: # هذا الشرط ربما يحتاج لإعادة نظر إذا كانت emb لا تزال ثلاثية الأبعاد هنا
                emb = emb.squeeze(0) # إذا كانت بهذا الشكل [1, batch_size, dim]
        return emb


def load_model(repo_id, dim="256"): # غيرنا repo_path_or_name إلى repo_id للتوضيح
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # اسم النموذج الأساسي (base model)
    base_model_name = "aubmindlab/bert-base-arabertv02"

    # 1. تحميل الـ Tokenizer من المستودع الخاص بك (Abdalrahmankamel/matryoshka-arabert)
    # هذا سيضمن تحميل ملفات tokenizer الخاصة بك إذا كنت قد قمت بتغييرها
    # وإلا، يمكن تحميله من base_model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
    except Exception:
        # إذا لم يكن الـ tokenizer الخاص بك متاحًا بشكل مباشر من repo_id
        # فارجع لتحميل tokenizer النموذج الأساسي
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)


    # 2. تحميل النموذج الأساسي (Base Model)
    base_model = AutoModel.from_pretrained(base_model_name).to(device)

    # 3. تحميل PEFT Adapter ودمجه مع النموذج الأساسي
    # PeftModel.from_pretrained يمكنه تحميل الـ adapter من repo_id
    peft_model = PeftModel.from_pretrained(base_model, repo_id).to(device)

    # 4. تهيئة MatryoshkaWrapper
    model = MatryoshkaWrapper(peft_model).to(device)

    try:
        from huggingface_hub import hf_hub_download
        # حاول تنزيل ملف matryoshka_wrapper.pt
        wrapper_weights_path = hf_hub_download(repo_id=repo_id, filename="matryoshka_wrapper.pt")
        state_dict = torch.load(wrapper_weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded matryoshka_wrapper.pt from {repo_id}")
    except Exception as e:
        print(f"Could not load matryoshka_wrapper.pt from Hugging Face Hub: {e}")
        # هذا يعني أن الـ wrapper_weights لم يتم تحميلها، مما قد يؤثر على النتائج

    model.eval()
    return model, tokenizer

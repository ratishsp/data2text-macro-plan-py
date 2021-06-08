"""
Classifies the ordinal adjective as inning or not
"""
import torch
import numpy as np
import logging

from transformers import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer)
}


def sample_sequence(model, length, context, num_samples=1, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :]
            probs, next_token = torch.topk(next_token_logits, 10, dim=-1)
            # print("probs", (F.softmax(probs)*100).tolist())
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def get_next_token(device, model, tokenizer, context):
    length = 1  # generate 1 token
    num_samples = 1

    raw_text = context
    context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
    out = sample_sequence(model=model, length=length, context=context_tokens, num_samples=num_samples, device=device)
    out = out[:, len(context_tokens):].tolist()
    candidate_outputs = []
    for o in out[0]:
        text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
        candidate_outputs.append(text.strip())  # strip out leading space
    return candidate_outputs


def load_model():
    seed = 42
    set_seed(seed, n_gpu=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = "gpt2"
    model_name_or_path = "gpt2-medium"
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    return device, model, tokenizer

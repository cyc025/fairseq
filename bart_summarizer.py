

import torch
import torchvision.models as models
from torch.autograd.profiler import profile, record_function

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')




def inference(max_length):
    # Generate Summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=max_length, truncation=True, early_stopping=True)
    # print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])


# open file for profiling
profile_log = open('profile.log','a')


import re
def extract_time(s):
    result = re.search(' (.*)ms', s)
    return (result.group(1).split('        ')[-1])

for ml in range(2,500):
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with profile(use_cuda=False) as prof:
        with record_function("model_inference"):
            inference(ml)
    profile_log.write(extract_time(str(prof))+'\n')

profile_log.close()

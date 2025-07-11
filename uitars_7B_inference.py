from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import torch
from tqdm import tqdm
# Load the model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "bytedance-research/UI-TARS-1.5-7B", torch_dtype=torch.bfloat16, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(
    "bytedance-research/UI-TARS-1.5-7B", padding_side="left"
)

model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

# Define the system prompt
## Below is the prompt for mobile
prompt = r"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```\nThought: ...
Action: ...\n```

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
type(content='')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
press_home()
press_back()
finished(content='') # Submit the task regardless of whether it succeeds or fails.

## Note
- Use English in `Thought` part.

- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""

with open("exploreBenchmark_width_w_high_quality.json", 'r') as f:
    data = json.load(f)

fout = open("exploreBenchmark_width_w_high_quality.jsonl", 'w', encoding='utf-8')
split_inference = 1
for cnt in tqdm(range((len(data)-1) // split_inference + 1)):
    inference_buffer = data[cnt*split_inference:(cnt+1)*split_inference]
    texts = []
    image_inputs = []
    for item in inference_buffer:
        img_filename = f"androidcontrol/{item['img_filename']}"
        # Define the input message
        # for gpt_output in item["gpt_output_split"]:
        high_level_instruction = item["high_level_instruction"]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": prompt + high_level_instruction,
                    },
                    {
                        "type": "image",
                        "image": img_filename,
                    }
                ],
            }
        ]

        # Prepare the input for the model
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_input, video_input = process_vision_info(messages)
        texts.append(text)
        image_inputs.append(image_input)
        
    inputs = processor(
        text=texts,
        images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=512)

    # Post-process the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert len(inference_buffer) == len(output_text)
    for input_item, output in zip(inference_buffer, output_text):
        input_item["uitars"] = output
    # print(output_text)
        fout.write(json.dumps(input_item, ensure_ascii=False) +'\n')

    # ['actions:\nCLICK <point>[[493,544]]</point><|im_end|>']

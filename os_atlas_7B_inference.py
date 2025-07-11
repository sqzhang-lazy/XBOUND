from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm
# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "OS-Copilot/OS-Atlas-Pro-7B", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "OS-Copilot/OS-Atlas-Pro-7B", padding_side="left"
)

# Define the system prompt
sys_prompt = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
       
Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]
    
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.
Custom Action 1: LONG_PRESS 
    - purpose: Long press at the specified position.
    - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>
    - example usage: LONG_PRESS <point>[[101, 872]]</point>
       
Custom Action 2: OPEN_APP
    - purpose: Open the specified application.
    - format: OPEN_APP [app_name]
    - example usage: OPEN_APP [Google Chrome]

Custom Action 3: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 4: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 5: PRESS_RECENT
    - purpose: Press the recent button to view or switch between recently used applications.
    - format: PRESS_RECENT
    - example usage: PRESS_RECENT

Custom Action 6: ENTER
    - purpose: Press the enter button.
    - format: ENTER
    - example usage: ENTER

Custom Action 7: WAIT
    - purpose: Wait for the screen to load.
    - format: WAIT
    - example usage: WAIT

Custom Action 8: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.
Thoughts: Clearly outline your reasoning process for current step.
Actions: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 

Your current task instruction, action history, and associated screenshot are as follows:
Screenshot: 
"""
with open("exploreBenchmark_width_w_high_quality.json.json", 'r') as f:
    data = json.load(f)

fout = open("exploreBenchmark_gpt4omini_osatlas_7B_Pro_output.jsonl", 'w', encoding='utf-8')
split_inference = 3
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
                        "type": "text", "text": sys_prompt,
                    },
                    {
                        "type": "image",
                        "image": img_filename,
                    },
                    {"type": "text", "text": f"Task instruction: {high_level_instruction}\nHistory: null" },
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
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Post-process the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert len(inference_buffer) == len(output_text)
    for input_item, output in zip(inference_buffer, output_text):
        input_item["osatlas"] = output
    # print(output_text)
        fout.write(json.dumps(input_item, ensure_ascii=False) +'\n')

    # ['actions:\nCLICK <point>[[493,544]]</point><|im_end|>']

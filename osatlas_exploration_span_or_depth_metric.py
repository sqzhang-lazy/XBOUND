import json
from collections import defaultdict
import re
import math
import copy

def main():
    json_file = ""
    depth_metric(json_file)

def is_coord_inside_box(x, y, box, width, height):
    left, top, right, bottom = box
    x *= (width / 1000)
    y *= (height / 1000)
    return left <= x <= right and top <= y <= bottom

def cal_coord_distance(x, y, golden_x, golden_y, width, height):
    golden_x /= width / 1000
    golden_y /= height / 1000
    return math.sqrt((golden_x - x)**2 + (golden_y - y)**2) <= 0.14 * 1000

def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    precision = len(common_tokens) / len(predicted_tokens) if predicted_tokens else 0
    recall = len(common_tokens) / len(ground_truth_tokens) if ground_truth_tokens else 0
    return 2 * (precision * recall) / (precision + recall) if precision + recall else 0


def original_acc_metric(json_path):
    data = [json.loads(ln) for ln in open(json_path, encoding='utf-8')]

    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    total_correct = 0

    for entry in data:
        try:
            if entry.get("osatlas") is not None:
                os_action = entry.get("osatlas").split("actions:\n")[1].split("<|im_end|>")[0]
            else:
                continue
        except:
            continue
        gt_action = entry.get("action")
        gt_action_type = gt_action["action_type"]
        os_action_type = os_action.split()[0]

        stats[gt_action_type]["total"] += 1

        if gt_action_type == "click" and os_action_type == "CLICK":
            try:
                pred_x, pred_y = map(int, re.findall(r'\d+', os_action)[:2])
            except:
                pred_x, pred_y = -100, -100
            golden_x = gt_action["x"]
            golden_y = gt_action["y"]
            if cal_coord_distance(pred_x, pred_y, golden_x, golden_y, entry.get("width"), entry.get("height")):
                total_correct += 1
                stats["click"]["correct"] += 1

        elif gt_action_type == "input_text" and os_action_type == "TYPE":
            if calculate_f1_score(os_action.split()[1].strip('[').strip(']'), gt_action["text"]) > 0.5:
                total_correct += 1
                stats["input_text"]["correct"] += 1

        elif gt_action_type == "scroll" and os_action_type == "SCROLL":
            if gt_action["direction"] in os_action.split()[1].lower():
                total_correct += 1
                stats["scroll"]["correct"] += 1

        elif gt_action_type == "long_press" and os_action_type == "LONG_PRESS":
            try:
                pred_x, pred_y = map(int, re.findall(r'\d+', os_action)[:2])
            except:
                pred_x, pred_y = -100, -100
            golden_x = gt_action["x"]
            golden_y = gt_action["y"]
            if cal_coord_distance(pred_x, pred_y, golden_x, golden_y, entry.get("width"), entry.get("height")):
                total_correct += 1
                stats["long_press"]["correct"] += 1

        elif gt_action_type == "open_app" and os_action_type == "OPEN_APP":
            if calculate_f1_score(os_action.split()[1].strip('[').strip(']'), gt_action["app_name"]) > 0.5:
                total_correct += 1
                stats["open_app"]["correct"] += 1

        elif gt_action_type == "navigate_home" and os_action_type == "PRESS_HOME":
            total_correct += 1
            stats["navigate_home"]["correct"] += 1

        elif gt_action_type == "navigate_back" and os_action_type == "PRESS_BACK":
            total_correct += 1
            stats["navigate_back"]["correct"] += 1

        elif gt_action_type == "wait" and os_action_type == "WAIT":
            total_correct += 1
            stats["wait"]["correct"] += 1


    total_acc = total_correct / len(data)

    print("\n=== Original & Total Accuracy ===")
    print(f"Total Number: {len(data)}")
    print("Total Accuracy: {:.2f}%".format(total_acc * 100))

    print("\n=== Action-wise Accuracy Summary ===")
    for action in ["click", "input_text", "scroll", "long_press", "open_app", "navigate_home", "navigate_back", "wait"]:
        correct = stats[action]["correct"]
        total = stats[action]["total"]
        acc = correct / total if total else 0
        print(f"{action.upper():<15} | Correct: {correct:<4} / Total: {total:<4} | Accuracy: {acc:.2%}")

def span_metric(json_path):
    data = [json.loads(ln) for ln in open(json_path, encoding='utf-8')]
    data = [item for item in data if item["action"]["action_type"] != "open_app" and item["action"]["action_type"] != "wait"]
    grouped_by_image = defaultdict(list)
    for item in data:
        img_filename = item.get("img_filename")
        if img_filename:
            grouped_by_image[img_filename].append(item)

    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    span_acc = 0
    total_correct = 0
    img_num = 0

    Learning_stage = 0
    Improvement_stage = 0
    Proficient_stage = 0
    Expert_stage = 0

    type_input_text_wrong = 0

    wrong_action_list = []
    explore_metric = []

    for img_file, entries in grouped_by_image.items():
        img_num += 1
        step_acc = 0
        tmp_wrong_action_list = []
        for entry in entries:
            try:
                if entry.get("osatlas") is not None:
                    os_action = entry.get("osatlas").split("actions:\n")[1].split("<|im_end|>")[0]
                else:
                    continue
            except:
                continue
            gt_action = entry.get("action")
            gt_action_type = gt_action["action_type"]
            os_action_type = os_action.split()[0]

            stats[gt_action_type]["total"] += 1

            if gt_action_type == "click" and os_action_type == "CLICK":
                try:
                    pred_x, pred_y = map(int, re.findall(r'\d+', os_action)[:2])
                except:
                    pred_x, pred_y = -100, -100
                judge_coord = is_coord_inside_box(pred_x, pred_y, gt_action["location"], entry.get("width"), entry.get("height"))
                if judge_coord is True:
                    step_acc += 1
                    stats["click"]["correct"] += 1
                else:
                    tmp_wrong_action_list.append(entry)
        

            elif gt_action_type == "type" and os_action_type == "TYPE":
                if calculate_f1_score(os_action.split("TYPE ")[1].strip('[').strip(']'), gt_action["text"]) > 0.5:
                    step_acc += 1
                    stats["type"]["correct"] += 1
                else:
                    tmp_wrong_action_list.append(entry)
                    type_input_text_wrong += 1

            elif gt_action_type == "scroll" and os_action_type == "SCROLL":
                if gt_action["direction"] in os_action.split()[1].lower():
                    step_acc += 1
                    stats["scroll"]["correct"] += 1
                else:
                    tmp_wrong_action_list.append(entry)

            elif gt_action_type == "long_press" and os_action_type == "LONG_PRESS":
                try:
                    pred_x, pred_y = map(int, re.findall(r'\d+', os_action)[:2])
                except:
                    pred_x, pred_y = -100, -100
                if is_coord_inside_box(pred_x, pred_y, gt_action["location"], entry.get("width"), entry.get("height")):
                    step_acc += 1
                    stats["long_press"]["correct"] += 1
                else:
                    tmp_wrong_action_list.append(entry)

            elif gt_action_type == "open_app" and os_action_type == "OPEN_APP":
                if calculate_f1_score(os_action.split()[1].strip('[').strip(']'), gt_action["app_name"]) > 0.5:
                    step_acc += 1
                    stats["open_app"]["correct"] += 1
                else:
                    tmp_wrong_action_list.append(entry)

            elif gt_action_type == "navigate_home" and os_action_type == "PRESS_HOME":
                step_acc += 1
                stats["navigate_home"]["correct"] += 1
            

            elif gt_action_type == "navigate_back" and os_action_type == "PRESS_BACK":
                step_acc += 1
                stats["navigate_back"]["correct"] += 1

            elif gt_action_type == "wait" and os_action_type == "WAIT":
                step_acc += 1
                stats["wait"]["correct"] += 1
            else:
                tmp_wrong_action_list.append(entry)

        if step_acc / len(entries) < 0.3:
            Learning_stage += 1
            wrong_action_list += copy.deepcopy(tmp_wrong_action_list)
        elif step_acc / len(entries) < 0.6:
            Improvement_stage += 1
        elif step_acc / len(entries) < 0.9:
            Proficient_stage += 1
        else:
            Expert_stage += 1
        explore_metric.append(step_acc / len(entries))
        span_acc += step_acc / len(entries)
        total_correct += step_acc

    span_acc /= img_num
    total_acc = total_correct / len(data)
    print(f"The text of Type Input is wrong: {type_input_text_wrong}")

    print("\n=== Span & Total Accuracy ===")
    print(f"Total Number: {len(data)}")
    print("Span Accuracy: {:.2f}%".format(span_acc * 100))
    print("Total Accuracy: {:.2f}%".format(total_acc * 100))

    print("\n=== Action-wise Accuracy Summary ===")
    for action in ["click", "type", "scroll", "long_press", "open_app", "navigate_home", "navigate_back", "wait"]:
        correct = stats[action]["correct"]
        total = stats[action]["total"]
        acc = correct / total if total else 0
        print(f"{action.upper():<15} | Correct: {correct:<4} / Total: {total:<4} | Accuracy: {acc:.2%}")
    
    print("\n=== Step-wise Accuracy Summary ===")
    print(f"The number and proportion of UI agents in the Learning Stage:: {Learning_stage} | Percentage: {Learning_stage/img_num:.2%}")
    print(f"The number and proportion of UI agents in the Improvement Stage:: {Improvement_stage} | Percentage: {Improvement_stage/img_num:.2%}")
    print(f"The number and proportion of UI agents in the Proficient Stage:: {Proficient_stage} | Percentage: {Proficient_stage/img_num:.2%}")
    print(f"The number and proportion of UI agents in the Expert Stage:: {Expert_stage} | Percentage: {Expert_stage/img_num:.2%}")


def depth_metric(json_path):
    data = [json.loads(ln) for ln in open(json_path, encoding='utf-8')]
    data = [item for item in data if item["action"]["action_type"] != "open_app" and item["action"]["action_type"] != "wait"]
    grouped_by_image = defaultdict(list)
    for item in data:
        img_filename = item.get("img_filename")
        if img_filename:
            grouped_by_image[img_filename].append(item)

    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    depth_acc = 0
    total_correct = 0
    img_num = 0

    Learning_stage = 0
    Improvement_stage = 0
    Proficient_stage = 0
    Expert_stage = 0

    wrong_action_list = []
    explore_metric = []

    for img_file, entries in grouped_by_image.items():
        img_num += 1
        step_acc = 0
        tmp_wrong_action_list = []
        for entry in entries:
            try:
                if entry.get("osatlas") is not None:
                    os_action = entry.get("osatlas").split("actions:\n")[1].split("<|im_end|>")[0]
                else:
                    continue
            except:
                continue
            gt_action = entry.get("action")
            gt_action_type = gt_action["action_type"]
            os_action_type = os_action.split()[0]
            stats[gt_action_type]["total"] += 1

            if gt_action_type == "click" and os_action_type == "CLICK":
                try:
                    pred_x, pred_y = map(int, re.findall(r'\d+', os_action)[:2])
                except:
                    pred_x, pred_y = -100, -100
                golden_x = gt_action["x"]
                golden_y = gt_action["y"]
                if cal_coord_distance(pred_x, pred_y, golden_x, golden_y, entry.get("width"), entry.get("height")):
                    step_acc += 1
                    stats["click"]["correct"] += 1
                else:
                    tmp_wrong_action_list.append(entry)

            elif gt_action_type == "input_text" and os_action_type == "TYPE":
                if calculate_f1_score(os_action.split("TYPE ")[1].strip('[').strip(']'), gt_action["text"]) > 0.5:
                    step_acc += 1
                    stats["input_text"]["correct"] += 1
                else:
                    tmp_wrong_action_list.append(entry)

            elif gt_action_type == "scroll" and os_action_type == "SCROLL":
                if gt_action["direction"] in os_action.split()[1].lower():
                    step_acc += 1
                    stats["scroll"]["correct"] += 1
                else:
                    tmp_wrong_action_list.append(entry)

            elif gt_action_type == "long_press" and os_action_type == "LONG_PRESS":
                try:
                    pred_x, pred_y = map(int, re.findall(r'\d+', os_action)[:2])
                except:
                    pred_x, pred_y = -100, -100
                golden_x = gt_action["x"]
                golden_y = gt_action["y"]
                if cal_coord_distance(pred_x, pred_y, golden_x, golden_y, entry.get("width"), entry.get("height")):
                    step_acc += 1
                    stats["long_press"]["correct"] += 1
                else:
                    tmp_wrong_action_list.append(entry)

            elif gt_action_type == "open_app" and os_action_type == "OPEN_APP":
                if calculate_f1_score(os_action.split()[1].strip('[').strip(']'), gt_action["app_name"]) > 0.5:
                    step_acc += 1
                    stats["open_app"]["correct"] += 1

            elif gt_action_type == "navigate_home" and os_action_type == "PRESS_HOME":
                step_acc += 1
                stats["navigate_home"]["correct"] += 1

            elif gt_action_type == "navigate_back" and os_action_type == "PRESS_BACK":
                step_acc += 1
                stats["navigate_back"]["correct"] += 1

            elif gt_action_type == "wait" and os_action_type == "WAIT":
                step_acc += 1
                stats["wait"]["correct"] += 1

            else:
                tmp_wrong_action_list.append(entry)
        if step_acc / len(entries) < 0.3:
            Learning_stage += 1
        elif step_acc / len(entries) < 0.6:
            Improvement_stage += 1
            wrong_action_list += copy.deepcopy(tmp_wrong_action_list)
        elif step_acc / len(entries) < 0.9:
            Proficient_stage += 1
        else:
            Expert_stage += 1
        explore_metric.append(step_acc / len(entries))
        depth_acc += step_acc / len(entries)
        total_correct += step_acc

    depth_acc /= img_num
    total_acc = total_correct / len(data)
    print("\n=== Depth & Total Accuracy ===")
    print(f"Total Number: {len(data)}")
    print("Depth Accuracy: {:.2f}%".format(depth_acc * 100))
    print("Total Accuracy: {:.2f}%".format(total_acc * 100))

    print("\n=== Action-wise Accuracy Summary ===")
    for action in ["click", "input_text", "scroll", "long_press", "open_app", "navigate_home", "navigate_back", "wait"]:
        correct = stats[action]["correct"]
        total = stats[action]["total"]
        acc = correct / total if total else 0
        print(f"{action.upper():<15} | Correct: {correct:<4} / Total: {total:<4} | Accuracy: {acc:.2%}")

    print("\n=== Step-wise Accuracy Summary ===")
    print(f"The number and proportion of UI agents in the Learning Stage:: {Learning_stage} | Percentage: {Learning_stage/img_num:.2%}")
    print(f"The number and proportion of UI agents in the Improvement Stage:: {Improvement_stage} | Percentage: {Improvement_stage/img_num:.2%}")
    print(f"The number and proportion of UI agents in the Proficient Stage:: {Proficient_stage} | Percentage: {Proficient_stage/img_num:.2%}")
    print(f"The number and proportion of UI agents in the Expert Stage:: {Expert_stage} | Percentage: {Expert_stage/img_num:.2%}")


if __name__ == "__main__":
    main()
    

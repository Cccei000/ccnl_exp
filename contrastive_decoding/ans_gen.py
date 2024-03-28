import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm
from time import time
import jsonlines
import argparse
import os


def load_queries(path, use_feedback):
    file = open(path, encoding='utf-8')
    data = list(jsonlines.Reader(file))
    file.close()
    if use_feedback is None:
        # load from e.g. output-13b-model0903-step15000-step15000-test1.json
        return [{'question': each["prompt"][0]} for each in data]
    else:
        # load from e.g. feedback_results.json
        return [{
            'question': each['question'],
            'answer': each['answer'],
            'feedback': each['feedback']} for each in data]

def prepare_prompt(tokenizer, query, use_feedback):
    if use_feedback == "conversation":
        '''
        conversation v1: append 请根据这些建议，重新回答问题，不要说多余的话。
        conversation v2: append 请根据这些建议，重新回答问题。
        conversation v3：append 再思考一下，然后重新回答问题。注意请直接给出回答。
        conversation v4：再思考一下，然后重新回答问题。注意请直接给出回答。

        '''
        # feedback = query['feedback'].strip()
        # if feedback.endswith('。'):
        #     feedback += '再思考一下，然后重新回答问题。注意请直接给出回答。'
        # else:
        #     feedback += '。再思考一下，然后重新回答问题。注意请直接给出回答。'
        feedback = '再思考一下，然后重新回答问题。注意请直接给出回答。'

        prompt = f"<human>:{query['question']}\n<bot>:{query['answer']}\n<human>:{feedback}\n<bot>:"
    elif use_feedback == "prompt":
        raise NotImplementedError
    elif use_feedback is None:
        prompt = f"<human>:{query['question']}\n<bot>:"
    
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    prompt = tokenizer.batch_decode(input_ids)[0]
    return prompt, input_ids

def process_generation(prompt, gen):
    assert gen.startswith(prompt)
    answer = gen[len(prompt):]
    answer = answer.replace("</s>", "").strip()
    return answer

def generate_answers(args):

    output_file_path = os.path.dirname(args.output_file)
    os.makedirs(output_file_path, exist_ok=True)

    print("Loading model...")
    start = time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).half().cuda().eval()
    print(f'time elapsed {time() - start:.5f} seconds')

    data = load_queries(args.query_file, args.use_feedback)
    print(f"Load {len(data)} lines of questions")

    output_data = []
    output_texts = []
    for query in tqdm(data):
        prompt, input_ids = prepare_prompt(tokenizer, query, args.use_feedback)
        outputs = model.generate(input_ids.cuda(),
                                 do_sample=True,
                                 temperature=0.85,
                                 top_p=0.95,
                                 max_length=args.max_length,
                                 repetition_penalty=1.)
        generation = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        if not generation.endswith("</s>") and not args.early_stop:
            continue
        answer = process_generation(prompt, generation)

        js_line = {
            "prompt": [query["question"]],
            "predict": answer,
            "use_feedback": args.use_feedback,
            "feedback": query.get("feedback")
            }
        output_data.append(js_line)
        output_texts.append("### Question: " + query["question"])
        output_texts.append("### Prompt: " + prompt)
        output_texts.append("### Asnwer: " + answer)
        output_texts.append("-" * 100)

    with jsonlines.open(args.output_file, "w") as file:
        for item in output_data:
            file.write(item)

    if args.output_file.endswith(".json"):
        with open(args.output_file.replace(".json", ".txt"), "w+") as f:
            f.write("\n".join(output_texts))
    elif args.output_file.endswith(".jsonl"):
        with open(args.output_file.replace(".jsonl", ".txt"), "w+") as f:
            f.write("\n".join(output_texts))        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--use_feedback", type=str.lower, default=None)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--early_stop", default=False, action="store_true")
    args = parser.parse_args()

    print(args.model_path)
    print(args.query_file)
    print(args.output_file)
    print(f'use_feedback {args.use_feedback}')
    generate_answers(args)
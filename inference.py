import sys
import os
import copy
import json
import argparse
import warnings
import torch

from PIL import Image
from tqdm import tqdm
from transformers import set_seed

from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLaVA on MathVerse")
    # Paths and identifiers
    parser.add_argument(
        "--pretrained",
        type=str,
        default="/home/shhj1998/checkpoints/LLaVA/models/llavanext-open_clip_hub:ViT-L-14-336-meta-llama_Meta-Llama-3-8B-Instruct-GeoCLIP_DAv2-gps-program-final-2",
        help="Path or hub id to the pretrained LLaVA checkpoint",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model identifier",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llava_llama_lora",
        help="Model name for conversation template selection",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default="/home/shhj1998/datasets/MathVerse/testmini.json",
        help="Path to MathVerse test JSON file",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/home/shhj1998/datasets/MathVerse/images",
        help="Directory containing images referenced by the JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.jsonl",
        help="Path to output file (JSON array by default)",
    )

    # Runtime options
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--device-map", type=str, default="auto", help="Device map for model")
    parser.add_argument("--conv-template", type=str, default="llava_llama_3", help="Conversation template key")
    parser.add_argument("--subject", type=str, default="Plane Geometry", help="Required subject filter if metadata present")
    parser.add_argument(
        "--exclude-problem-version",
        type=str,
        default="Vision Only",
        help="Exclude rows whose problem_version equals this value",
    )
    parser.add_argument("--seed", type=int, default=7777, help="Random seed")

    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--num-return-sequences", type=int, default=10)
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (disabled by default for beam search)",
    )

    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    # Load model and tokenizer
    llava_model_args = {"multimodal": True, "attn_implementation": "sdpa"}
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        args.pretrained,
        args.model_base,
        args.model_name,
        device_map=args.device_map,
        **llava_model_args,
    )
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_250|>"})
    model.config.pad_token_id = tokenizer.pad_token_id  # updating model config
    tokenizer.padding_side = "right"
    model.eval()

    set_seed(args.seed)

    conv_template = args.conv_template  # Ensure correct chat template for the model
    with open(args.dataset_json, "r") as f:
        test_data = json.load(f)

    device = args.device

    with torch.no_grad():
        responses = []
        for row in tqdm(test_data):
            if ("metadata" in row and row["metadata"].get("subject") != args.subject) or row.get("problem_version") == args.exclude_problem_version:
                continue

            image_path = os.path.join(args.images_dir, row["image"])
            image = Image.open(image_path)
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
            image_sizes = [image.size]

            question = row["question"]
            if "Choices" in question:
                question = question.split("Choices")[0]
            question = DEFAULT_IMAGE_TOKEN + "\n" + question
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )[None].to(device)

            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=args.do_sample,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=args.num_return_sequences,
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            response = text_outputs
            responses.append(
                {
                    "image": row["image"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "response": response,
                    "problem_version": row.get("problem_version"),
                }
            )

        with open(args.output, "w") as f:
            json.dump(responses, f)


if __name__ == "__main__":
    main()

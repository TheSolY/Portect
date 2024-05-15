import argparse
from diffusers import DiffusionPipeline
import torch
import os


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Generate images to test Portect.")
    parser.add_argument(
        "--org_model_path",
        type=str,
        help="Path to the model trained on the original images.",
    )
    parser.add_argument(
        "--interp_model_path",
        type=str,
        help="Path to the model trained on the interpolated images.",
    )
    parser.add_argument(
        "--images_per_prompt",
        type=int,
        default=10,
        help="How many images to generate per prompt.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./generated_images",
        help="Directory to save generated images."
    )
    parser.add_argument(
        "--prompt_list",
        action="append",
        help="List of prompts to generate.",
        default=["DSLR photo of ukj wearing a white shirt, detailed face",
                 "ukj, portrait, natural light, bokeh",
                 "selfie of ukj on the beach",
                 "professional headshot of ukj in a suit"]
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use- CUDA or CPU.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.org_model_path is None and args.interp_model_path is None:
        raise ValueError("Specify at least one of: `--org_model_path` or `--interp_model_path`.")

    return args

num_images = 10
save_dir = '/tmp/sol/portect/gen_images_portect_org'


def main(args):
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    if args.org_model_path is not None:
        if not os.path.isdir(os.path.join(args.save_dir, "org")):
            os.mkdir(os.path.join(args.save_dir, "org"))

        pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
        pipe = pipe.to(args.device)
        pipe.load_lora_weights(args.org_model_path)

        for prompt in args.prompt_list:
            images = pipe(prompt,
                          num_inference_steps=100, guidance_scale=7.5, num_images_per_prompt=args.images_per_prompt)[0]
            for i in range(args.images_per_prompt):
                images[i].save(os.path.join(args.save_dir, "org", prompt + str(i) + ".png"))

    if args.interp_model_path is not None:
        if not os.path.isdir(os.path.join(args.save_dir, "interp")):
            os.mkdir(os.path.join(args.save_dir, "interp"))

        pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
        pipe = pipe.to(args.device)
        pipe.load_lora_weights(args.interp_model_path)

        for prompt in args.prompt_list:
            images = pipe(prompt,
                          num_inference_steps=100, guidance_scale=7.5, num_images_per_prompt=args.images_per_prompt)[0]
            for i in range(args.images_per_prompt):
                images[i].save(os.path.join(args.save_dir, "interp", prompt + str(i) + ".png"))


if __name__ == "__main__":
    args = parse_args()
    main(args)

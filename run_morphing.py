# T.B.C.

import os
import sys
import time
import subprocess
import argparse

from FILM import process_keyframes

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Orchestrate DiffMorpher || LCM-LoRa || LCM, and FILM for smooth morphing between two images.")
    
    # ------------------- KEYFRAME METHOD SELECTION -------------------
    # TODO multiple methods not supported yet
    # parser.add_argument(
    #     "--keyframe_method", type=str, default="diffmorpher",
    #     choices=["diffmorpher", "lcm_lora", "diffmorpher_lcm"],
    #     help="Choose which method to use for keyframe generation."
    # )
    # i think this would be to select the models. so theres
    # stable diffusion 2-1 (without lcm)
    # stable diffusion 1-5 (has lcm support)
    # dreamshaper 7, fine-tuned 1-5, more glam, (has lcm support)

    # ------------------- DIFFMORPHER ARGS -------------------
    parser.add_argument(
        "--model_path", type=str, default="stabilityai/stable-diffusion-2-1-base", # yeah this needs to be chnaged TODO
        help="Pretrained model to use for DiffMorpher (default: %(default)s)"
    )
    parser.add_argument(
        "--image_path_0", type=str, required=True,
        help="Path of the first image"
    )
    parser.add_argument(
        "--prompt_0", type=str, default="",
        help="Prompt describing the first image (default: %(default)s)"
    )
    parser.add_argument(
        "--image_path_1", type=str, required=True,
        help="Path of the second image"
    )
    parser.add_argument(
        "--prompt_1", type=str, default="",
        help="Prompt describing the second image (default: %(default)s)"
    )
    parser.add_argument(
        "--output_path", type=str, default="./results",
        help="Output folder for DiffMorpher keyframes/gif (default: %(default)s)"
    )
    parser.add_argument(
        "--save_lora_dir", type=str, default="./lora",
        help="Directory to save LoRA outputs (default: %(default)s)"
    )
    parser.add_argument(
        "--load_lora_path_0", type=str, default="",
        help="Path to LoRA checkpoint for image 0 (default: %(default)s)"
    )
    parser.add_argument(
        "--load_lora_path_1", type=str, default="",
        help="Path to LoRA checkpoint for image 1 (default: %(default)s)"
    )
    parser.add_argument(
        "--use_adain", action="store_true",
        help="Use AdaIN in DiffMorpher pipeline"
    )
    parser.add_argument(
        "--use_reschedule", action="store_true",
        help="Use reschedule sampling in DiffMorpher"
    )
    parser.add_argument(
        "--lamb", type=float, default=0.6,
        help="Lambda for self-attention replacement in DiffMorpher (default: %(default)s)"
    )
    parser.add_argument(
        "--fix_lora_value", type=float, default=None,
        help="Fix LoRA value in DiffMorpher (default: LoRA interpolation)"
    )
    parser.add_argument(
        "--save_inter", action="store_true",
        help="Save intermediate frames as individual images (e.g. .png) in DiffMorpher"
    )
    parser.add_argument(
        "--num_frames", type=int, default=16,
        help="Number of keyframes to generate (default: %(default)s)"
    )
    parser.add_argument(
        "--duration", type=int, default=100,
        help="Duration of each keyframe in the final .gif (default: %(default)s ms)"
    )
    parser.add_argument(
        "--no_lora", action="store_true",
        help="Disable LoRA usage in DiffMorpher"
    )

    # ------------------- FILM ARGS -------------------
    parser.add_argument(
        "--use_film", action="store_true",
        help="Flag to indicate whether to run FILM after generating keyframes"
    )
    parser.add_argument(
        "--film_input_folder", type=str, default="",
        help="Folder containing keyframes for FILM. If empty, will use DiffMorpher output folder."
    )
    parser.add_argument(
        "--film_output_folder", type=str, default="./FILM_Results",
        help="Folder where FILM’s final interpolated video is saved (default: %(default)s)"
    )
    parser.add_argument(
        "--film_fps", type=int, default=40,
        help="FPS for the final video - 'Pseudo-Playback-Speed', since total frames are same (default: %(default)s)"
    )
    parser.add_argument(
        "--film_num_recursions", type=int, default=3,
        help="Number of recursive interpolations to perform in FILM (default: %(default)s)"
    )

    return parser.parse_args()

def run_diffmorpher(args):
    """
    Calls DiffMorpher's main.py via subprocess using the CLI arguments.
    Expects `DiffMorpher/` to be a submodule in the current repo.
    """
    diffmorpher_script = os.path.join("DiffMorpher", "main.py")

    cmd = [
        sys.executable, diffmorpher_script,
        "--model_path", args.model_path,
        "--image_path_0", args.image_path_0,
        "--prompt_0", args.prompt_0,
        "--image_path_1", args.image_path_1,
        "--prompt_1", args.prompt_1,
        "--output_path", args.output_path,
        "--save_lora_dir", args.save_lora_dir,
        "--lamb", str(args.lamb),
        "--num_frames", str(args.num_frames),
        "--duration", str(args.duration),
    ]

    if args.load_lora_path_0:
        cmd += ["--load_lora_path_0", args.load_lora_path_0]
    if args.load_lora_path_1:
        cmd += ["--load_lora_path_1", args.load_lora_path_1]
    if args.use_adain:
        cmd.append("--use_adain")
    if args.use_reschedule:
        cmd.append("--use_reschedule")
    if args.fix_lora_value is not None:
        cmd += ["--fix_lora_value", str(args.fix_lora_value)]
    if args.save_inter:
        cmd.append("--save_inter")
    if args.no_lora:
        cmd.append("--no_lora")

    print("[INFO] Running DiffMorpher with command:")
    print(" ".join(cmd))

    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()
    print(f"[INFO] DiffMorpher completed in {end - start:.2f} seconds.")

def run_lcm_lora(args): 
    """
    Run the LCM-LoRA accelerated Keyframe generator

    """
    # THIS would be a flag passed to diffmorpher instead.
    pass


def create_simple_video_from_keyframes(keyframes_folder, output_folder, fps=40):
    """
    If the user does NOT want FILM, we still make a basic video from keyframes.
    Assumes frames are saved as .png or .jpg in keyframes_folder.
    """
    import cv2
    from glob import glob
    import os
    from datetime import datetime
    
    os.makedirs(output_folder, exist_ok=True)

    images = sorted(glob(os.path.join(keyframes_folder, "*.png")))
    if not images:
        images = sorted(glob(os.path.join(keyframes_folder, "*.jpg")))
    if not images:
        print(f"[WARN] No .png or .jpg frames found in {keyframes_folder}.")
        return

    # Prepare video writer
    first_frame = cv2.imread(images[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video_path = os.path.join(output_folder, f"simple_morph_{timestamp}.mp4")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"[INFO] Basic morphing video saved at: {out_video_path}")

def main():
    args = parse_arguments()
    overall_start_time = time.time()

    # 1) Run DiffMorpher to generate keyframes
    run_diffmorpher(args)

    # 2) Determine the folder containing the keyframes
    # If user didn’t explicitly give `--film_input_folder`, use `args.output_path`
    keyframes_folder = args.film_input_folder if args.film_input_folder else args.output_path

    # 3) If user wants to use FILM, perform high-quality interpolation on the keyframes
    if args.use_film:
        print("[INFO] Running FILM to enhance the keyframes...")

        start_film_time = time.time()
        # from FILM.py:
        process_keyframes(
            input_folder=keyframes_folder,
            output_folder=args.film_output_folder,
            fps=args.film_fps,
            num_recursions=args.film_num_recursions
        )
        end_film_time = time.time()
        print(f"[INFO] FILM interpolation completed in {end_film_time - start_film_time:.2f} seconds.")

    else:
        # 4) If user does NOT want FILM, create a simple .mp4 from the keyframes
        print("[INFO] Skipping FILM interpolation. Creating a basic video from DiffMorpher keyframes...")
        create_simple_video_from_keyframes(
            keyframes_folder=keyframes_folder,
            output_folder=args.film_output_folder,
            fps=args.film_fps
        )

    # 5) Print total execution time
    overall_end_time = time.time()
    print(f"[INFO] Entire pipeline completed in {overall_end_time - overall_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
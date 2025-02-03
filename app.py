import os
import sys
import subprocess
import tempfile
import streamlit as st
from PIL import Image

def save_uploaded_file(uploaded_file, dst_path):
    """Save an uploaded file (Streamlit) to a destination path on disk."""
    with open(dst_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  

def main():
    st.title("DiffMorpher + LCM-LoRA + FILM Web App")
    st.markdown("""
    This application demonstrates an **image morphing** pipeline that combines:
    - **DiffMorpher** for keyframe generation,
    - **Latent Consistency Model** (LCM) based LoRA acceleration,
    - **FILM** for advanced frame interpolation,
    - All integrated into a single UI.

    Upload two images, enter prompts if desired, and configure advanced settings. 
    Then, generate a morphing video between them!
    """)

    # ---------------------
    # IMAGE & PROMPT INPUTS
    # ---------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image A")
        uploaded_image_A = st.file_uploader("Choose first image...", type=["png", "jpg", "jpeg"])
        prompt_A = st.text_input("Prompt describing Image A (optional)", value="")

    with col2:
        st.subheader("Image B")
        uploaded_image_B = st.file_uploader("Choose second image...", type=["png", "jpg", "jpeg"])
        prompt_B = st.text_input("Prompt describing Image B (optional)", value="")

    st.markdown("---")

    # ---------------------
    # ADVANCED SETTINGS
    # ---------------------
    with st.expander("Advanced Morph Settings"):
        num_frames = st.number_input("Number of keyframes to generate", min_value=2, max_value=200, value=16)
        fps = st.number_input("FPS of final video", min_value=1, max_value=120, value=40)
        film_interpolation = st.checkbox("Use FILM for advanced interpolation?", value=True)
        film_recursions = st.number_input("FILM recursive passes", min_value=1, max_value=6, value=3)
        do_lora = st.checkbox("Disable LoRA usage?", value=False)
        fix_lora_val = st.text_input("Fix LoRA alpha (optional, e.g. 0.7)", value="")
        # Add more advanced parameters as needed, e.g. lambda, guidance scale, etc.

    st.markdown("---")

    # ---------------------
    # MORPHING EXECUTION
    # ---------------------
    if st.button("Run Morphing Pipeline"):
        # Validate input
        if not (uploaded_image_A and uploaded_image_B):
            st.error("Please upload both images before running the morphing pipeline.")
            return

        # Save images to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            imgA_path = os.path.join(temp_dir, "imageA.png")
            imgB_path = os.path.join(temp_dir, "imageB.png")
            save_uploaded_file(uploaded_image_A, imgA_path)
            save_uploaded_file(uploaded_image_B, imgB_path)

            # OUTPUT folder for results
            output_dir = os.path.join(temp_dir, "morph_results")
            os.makedirs(output_dir, exist_ok=True)

            film_output_dir = os.path.join(temp_dir, "film_output")
            os.makedirs(film_output_dir, exist_ok=True)

            # Build the command for run_morphing.py
            cmd = [
                sys.executable, "run_morphing.py",
                "--image_path_0", imgA_path,
                "--image_path_1", imgB_path,
                "--prompt_0", prompt_A,
                "--prompt_1", prompt_B,
                "--output_path", output_dir,
                "--num_frames", str(num_frames),
                "--film_output_folder", film_output_dir,
                "--film_fps", str(fps),
                "--film_num_recursions", str(film_recursions),
            ]

            if film_interpolation:
                cmd.append("--use_film")
            if do_lora:
                cmd.append("--no_lora")
            if fix_lora_val:
                cmd.extend(["--fix_lora_value", fix_lora_val])

            st.info("Running morphing pipeline. Please wait...")
            with st.spinner("Generating morph..."):
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    st.error(f"Error while running pipeline: {e}")
                    return

            # Collect the final result (video)
            # The pipeline either produces a .mp4 in film_output_dir or a fallback .mp4 from basic creation
            possible_outputs = [f for f in os.listdir(film_output_dir) if f.endswith(".mp4")]
            if not possible_outputs:
                # might be in output_dir if FILM wasn't used
                possible_outputs = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]

            if possible_outputs:
                final_video_path = os.path.join(
                    film_output_dir if film_interpolation else output_dir,
                    possible_outputs[0]
                )
                st.success(f"Morphing complete! Found {final_video_path}")

                st.video(final_video_path)
                with open(final_video_path, "rb") as f:
                    st.download_button(
                        "Download Result Video",
                        data=f.read(),
                        file_name="morph_result.mp4",
                        mime="video/mp4"
                    )
            else:
                st.warning("No .mp4 output file found. Check logs or advanced settings for errors.")

if __name__ == "__main__":
    main()

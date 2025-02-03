import os
import sys
import subprocess
import tempfile
import base64
from io import BytesIO

import streamlit as st
from PIL import Image

# Set Streamlit page configuration
st.set_page_config(
    page_title="Metamorph: DiffMorpher + LCM-LoRA + FILM",
    layout="wide",
    page_icon="🌀"
)

def save_uploaded_file(uploaded_file, dst_path):
    with open(dst_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def get_img_as_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def main():
    # ---------------- CUSTOM CSS FOR A PROFESSIONAL, DARK THEME ----------------
    st.markdown(
        """
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        /* Global styling */
        body {
            font-family: 'Roboto', sans-serif;
            color: #f1f1f1;
        }
        h1, h2, h3 {
            color: #ffffff;
        }
        p, span, label {
            color: #f1f1f1;
        }
        body, p {
            line-height: 1.6;
            letter-spacing: 0.3px;
        }
        /* Header: Centered large logo and title */
        .header-logo-large {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 200px;
        }
        .header-title {
            text-align: center;
            font-size: 2.8rem;
            font-weight: bold;
            color: #ffffff;
            margin-top: 0.5rem;
        }
        /* Darker, slower animated background */
        .stApp {
            background: linear-gradient(315deg, #000428, #004e92);
            animation: gradient 30s ease infinite;
            background-size: 400% 400%;
            background-attachment: fixed;
        }
        @keyframes gradient {
            0% { background-position: 0% 0%; }
            50% { background-position: 100% 100%; }
            100% { background-position: 0% 0%; }
        }
        /* Main container styling */
        .main .block-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
            background-color: transparent;
            color: #f1f1f1;
        }
        /* Button styling */
        div.stButton > button {
            background-color: #4caf50;
            color: #ffffff;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 5px;
            cursor: pointer;
            font-family: 'Roboto', sans-serif;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #45a049;
        }
        /* File uploader label styling */
        .stFileUploader label {
            font-size: 1rem;
            color: #f1f1f1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---------------- HEADER & LOGO ----------------
    logo_path = os.path.join("lcm-lora", "metamorphLogo_nobg.png")
    if os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path)
            logo_base64 = get_img_as_base64(logo)
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{logo_base64}" class="header-logo-large" alt="Metamorph Logo">
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception:
            pass

    st.markdown("<h1 class='header-title'>Metamorph Web App</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align: center; font-size: 1.1rem;'>
            DiffMorpher is used for keyframe generation by default, with FILM for interpolation.
            Optionally, enable LCM-LoRA for accelerated processing (with a slight quality trade-off).
            Upload two images, optionally provide textual prompts, and fine-tune the settings to create a smooth, high-quality morphing video.
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )

    # ---------------- SECTION 1: IMAGE & PROMPT INPUTS ----------------
    st.subheader("1. Upload Source Images & Prompts")
    col_imgA, col_imgB = st.columns(2)
    with col_imgA:
        st.markdown("#### Image A")
        uploaded_image_A = st.file_uploader("Upload your first image (jpeg, png, etc.)", type=["png", "jpg", "jpeg"], key="imgA")
        prompt_A = st.text_input("Prompt for Image A (optional)", value="", key="promptA")
    with col_imgB:
        st.markdown("#### Image B")
        uploaded_image_B = st.file_uploader("Upload your second image (jpeg, png, etc.)", type=["png", "jpg", "jpeg"], key="imgB")
        prompt_B = st.text_input("Prompt for Image B (optional)", value="", key="promptB")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- SECTION 2: MORPHING SETTINGS ----------------
    st.subheader("2. Configure Morphing Settings")
    st.markdown(
        """
        By default, DiffMorpher is used for keyframe generation and FILM for interpolation.
        Optionally, you can enable LCM-LoRA for accelerated processing.
        """,
        unsafe_allow_html=True
    )
    # LCM-LoRA placeholder checkbox
    use_lcm_lora = st.checkbox("Enable LCM-LoRA (accelerated processing with slight quality trade-off)", value=False)

    # Advanced Options block (restored from earlier version)
    with st.expander("Advanced Options", expanded=True):
        col_left, col_mid, col_right = st.columns(3)
        with col_left:
            num_frames = st.number_input("Number of keyframes (2–200)", min_value=2, max_value=200, value=16)
            film_interpolation = st.checkbox("Use FILM interpolation", value=True)
            do_lora = st.checkbox("Disable LoRA usage? [--no_lora]", value=False)
        with col_mid:
            fps = st.number_input("FPS of final video (1–120)", min_value=1, max_value=120, value=40)
            film_recursions = st.number_input("FILM recursion passes (1–6)", min_value=1, max_value=6, value=3)
            fix_lora_val = st.text_input("Fix LoRA alpha (optional, e.g. '0.7')", value="")
        with col_right:
            use_adain = st.checkbox("Use AdaIN? [--use_adain]", value=False)
            use_reschedule = st.checkbox("Use reschedule sampling? [--use_reschedule]", value=False)
            save_inter = st.checkbox("Save intermediate frames? [--save_inter]", value=False)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- SECTION 3: EXECUTE MORPH PIPELINE ----------------
    st.subheader("3. Generate Morphing Video")
    st.markdown("Once satisfied with your inputs, click below to start the process.")
    if st.button("Run Morphing Pipeline", key="run_pipeline"):
        if not (uploaded_image_A and uploaded_image_B):
            st.error("Please upload both images before running the morphing pipeline.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            imgA_path = os.path.join(temp_dir, "imageA.png")
            imgB_path = os.path.join(temp_dir, "imageB.png")
            save_uploaded_file(uploaded_image_A, imgA_path)
            save_uploaded_file(uploaded_image_B, imgB_path)

            output_dir = os.path.join(temp_dir, "morph_results")
            film_output_dir = os.path.join(temp_dir, "film_output")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(film_output_dir, exist_ok=True)

            cmd = [
                sys.executable, "run_morphing.py",
                "--image_path_0", imgA_path,
                "--image_path_1", imgB_path,
                "--prompt_0", prompt_A,
                "--prompt_1", prompt_B,
                "--output_path", output_dir,
                "--film_output_folder", film_output_dir,
            ]
            if use_lcm_lora:
                cmd.append("--use_lcm_lora")
            if film_interpolation:
                cmd.append("--use_film")
            if do_lora:
                cmd.append("--no_lora")
            if fix_lora_val.strip():
                cmd.extend(["--fix_lora_value", fix_lora_val.strip()])
            if use_adain:
                cmd.append("--use_adain")
            if use_reschedule:
                cmd.append("--use_reschedule")
            if save_inter:
                cmd.append("--save_inter")

            st.info("Initializing pipeline. Please wait...")
            with st.spinner("Generating morph..."):
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    st.error(f"Error running pipeline: {e}")
                    return

            possible_outputs = [f for f in os.listdir(film_output_dir) if f.endswith(".mp4")]
            if not possible_outputs:
                possible_outputs = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]

            if possible_outputs:
                final_video_path = os.path.join(
                    film_output_dir if os.listdir(film_output_dir) else output_dir,
                    possible_outputs[0]
                )
                st.success("Morphing complete! 🎉")
                st.video(final_video_path)
                with open(final_video_path, "rb") as f:
                    st.download_button(
                        "Download Result Video",
                        data=f.read(),
                        file_name="morph_result.mp4",
                        mime="video/mp4"
                    )
            else:
                st.warning("No .mp4 output found. Check logs for details.")

if __name__ == "__main__":
    main()
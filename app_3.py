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
    # ---------------- CUSTOM CSS FOR VIBRANT BACKGROUND ----------------
    st.markdown(
        """
        <style>
        /* Target the main Streamlit container for background */
        .stApp {
            background: linear-gradient(315deg, #4f2991 3%, #7dc4ff 38%, #36cfcc 68%, #a92ed3 98%);
            animation: gradient 15s ease infinite;
            background-size: 400% 400%;
            background-attachment: fixed;
        }
        @keyframes gradient {
            0% { background-position: 0% 0%; }
            50% { background-position: 100% 100%; }
            100% { background-position: 0% 0%; }
        }
        /* Ensure the content container is transparent to show the gradient */
        .main .block-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
            background-color: transparent;
        }
        /* Header styling */
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .header-logo {
            flex-shrink: 0;
            width: 80px;
        }
        .header-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #ffffff;  /* White text to contrast with the vibrant gradient */
        }
        /* Subheader style */
        .subheader {
            color: #ffffff;
            margin-bottom: 1rem;
            border-bottom: 2px solid #ced4da;
            padding-bottom: 0.3rem;
        }
        /* Button styling */
        div.stButton > button {
            background-color: #4caf50;
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #45a049;
        }
        /* Download button styling */
        .stDownloadButton button {
            background-color: #4caf50;
            color: white;
            border: none;
            padding: 0.6rem 1rem;
            border-radius: 5px;
            cursor: pointer;
        }
        .stDownloadButton button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---------------- HEADER & LOGO ----------------
    logo_path = os.path.join("lcm-lora", "metamorphLogo_nobg.png")
    logo = None
    if os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path)
        except Exception:
            logo = None

    with st.container():
        if logo:
            logo_base64 = get_img_as_base64(logo)
            st.markdown(
                f"""
                <div class="header-container">
                    <img src="data:image/png;base64,{logo_base64}" class="header-logo" alt="Logo">
                    <div class="header-title">Metamorph Web App</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h1 class='header-title' style='text-align: center;'>Metamorph Web App</h1>",
                unsafe_allow_html=True
            )

    st.markdown(
        """
        <p style='text-align: center; font-size: 1.1rem; color: #f1f1f1;'>
            <strong>Metamorph</strong> seamlessly blends:
            <br>- <strong>DiffMorpher</strong> for keyframe generation,
            <br>- <strong>Latent Consistency Model</strong> (LCM) with LoRA for accelerated keyframing,
            <br>- <strong>FILM</strong> for advanced frame interpolation.
            <br><br>
            Upload two images, optionally provide textual prompts, and fine-tune
            the settings to create a smooth, high-quality morphing video.
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )

    # ---------------- SECTION 1: IMAGE & PROMPT INPUTS ----------------
    st.markdown("<h3 class='subheader'>1. Upload Source Images & Prompts</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p>Provide the two images you want to morph, along with optional prompts for each image.</p>",
        unsafe_allow_html=True
    )

    col_imgA, col_imgB = st.columns(2)
    with col_imgA:
        st.markdown("#### Image A")
        uploaded_image_A = st.file_uploader("Upload your first image", type=["png", "jpg", "jpeg"], key="imgA")
        prompt_A = st.text_input("Prompt for Image A (optional)", value="", key="promptA")
    with col_imgB:
        st.markdown("#### Image B")
        uploaded_image_B = st.file_uploader("Upload your second image", type=["png", "jpg", "jpeg"], key="imgB")
        prompt_B = st.text_input("Prompt for Image B (optional)", value="", key="promptB")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- SECTION 2: ADVANCED SETTINGS ----------------
    st.markdown("<h3 class='subheader'>2. Configure Morphing Settings</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p>Toggle additional features or fine-tune generation details below.</p>",
        unsafe_allow_html=True
    )
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
    st.markdown("<h3 class='subheader'>3. Generate Morphing Video</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p>Once satisfied with your inputs, click below to start the process.</p>",
        unsafe_allow_html=True
    )
    if st.button("Run Morphing Pipeline", key="run_pipeline"):
        # Validate image uploads
        if not (uploaded_image_A and uploaded_image_B):
            st.error("Please upload both images before running the morphing pipeline.")
            return

        # Save uploaded images to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            imgA_path = os.path.join(temp_dir, "imageA.png")
            imgB_path = os.path.join(temp_dir, "imageB.png")
            save_uploaded_file(uploaded_image_A, imgA_path)
            save_uploaded_file(uploaded_image_B, imgB_path)

            # Prepare output directories
            output_dir = os.path.join(temp_dir, "morph_results")
            film_output_dir = os.path.join(temp_dir, "film_output")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(film_output_dir, exist_ok=True)

            # Build CLI command for run_morphing.py
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

            # Check for output video (.mp4)
            possible_outputs = [f for f in os.listdir(film_output_dir) if f.endswith(".mp4")]
            if not possible_outputs:
                possible_outputs = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]

            if possible_outputs:
                final_video_path = os.path.join(
                    film_output_dir if film_interpolation else output_dir,
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
"""
Cleaned up version, Close-to-Final UI features and functionality logic. 
"""
import os
import sys
import subprocess
import tempfile
import base64
from io import BytesIO

import streamlit as st
from PIL import Image

# Set Streamlit page configuration (centered content via CSS)
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
        h1, h2, h3, h4 {
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
        /* Dark animated background */
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
        /* Run button styling */
        div.stButton > button {
            background-image: linear-gradient(45deg, #8e44ad, #732d91);
            box-shadow: 0 0 10px rgba(142,68,173,0.6), 0 0 20px rgba(114,45,145,0.4);
            border: none;
            color: #ffffff;
            padding: 0.6rem 1.2rem;
            border-radius: 5px;
            cursor: pointer;
            font-family: 'Roboto', sans-serif;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        div.stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 20px rgba(142,68,173,0.8), 0 0 30px rgba(114,45,145,0.6);
        }
        /* File uploader label styling */
        .stFileUploader label {
            font-size: 1rem;
            color: #f1f1f1;
        }
        /* Advanced Options divider styling */
        .right-column-divider {
            border-left: 2px solid #f1f1f1;
            padding-left: 1rem;
            margin-left: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---------------- HEADER & LOGO ----------------
    logo_path = "metamorphLogo_nobg.png"
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
            Optionally, you can enable LCM-LoRA for accelerated inference (with slight decrease in quality).
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
        uploaded_image_A = st.file_uploader("Upload your first image", type=["png", "jpg", "jpeg"], key="imgA")
        if uploaded_image_A is not None:
            # use_container_width instead of use_column_width
            st.image(uploaded_image_A, caption="Preview - Image A", use_container_width=True)
        prompt_A = st.text_input("Prompt for Image A (optional)", value="", key="promptA")
    with col_imgB:
        st.markdown("#### Image B")
        uploaded_image_B = st.file_uploader("Upload your second image", type=["png", "jpg", "jpeg"], key="imgB")
        if uploaded_image_B is not None:
            # use_container_width instead of use_column_width
            st.image(uploaded_image_B, caption="Preview - Image B", use_container_width=True)
        prompt_B = st.text_input("Prompt for Image B (optional)", value="", key="promptB")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- SECTION 2: CONFIGURE MORPHING PIPELINE ----------------
    st.subheader("2. Configure Morphing Pipeline")
    st.markdown(
        """
        <p style="font-size: 1rem;">
            Select a preset below to automatically adjust quality and inference time. 
            If you choose <strong>Custom ⚙️</strong>, the advanced settings will automatically expand so you can fine-tune the configuration.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Preset Options (Dropdown)
    st.markdown("**Preset Options**")
    preset_option = st.selectbox(
        "Select a preset for quality and inference time",
        options=[
            "Maximum quality, highest inference time 🏆",
            "Medium quality, medium inference time ⚖️",
            "Low quality, lowest inference time ⚡",
            "Creative morph 🎨",
            "Custom ⚙️"
        ],
        index=0,
        label_visibility="collapsed"  # Hide the label in the UI but keep it for accessibility
    )

    # Determine preset defaults based on selection
    if preset_option.startswith("Maximum quality"):
        # "Maximum quality, highest inference time 🏆"
        preset_model = "Base Stable Diffusion V2-1 (No LCM-LoRA support)"
        preset_film = True
        preset_lcm = False
    elif preset_option.startswith("Medium quality"):
        # "Medium quality, medium inference time ⚖️"
        preset_model = "Base Stable Diffusion V2-1 (No LCM-LoRA support)"
        preset_film = False
        preset_lcm = False
    elif preset_option.startswith("Low quality"):
        # "Low quality, lowest inference time ⚡"
        preset_model = "Base Stable Diffusion V1-5"
        preset_film = False
        preset_lcm = True
    elif preset_option.startswith("Creative morph"):
        # "Creative morph 🎨"
        preset_model = "Dreamshaper-7 (fine-tuned SD V1-5)"
        preset_film = True
        preset_lcm = True
    else:
        # "Custom ⚙️"
        preset_model = None
        preset_film = None
        preset_lcm = None

    # Auto-expand advanced options if "Custom ⚙️" is chosen
    advanced_expanded = True if preset_option.endswith("⚙️") else False

    # Advanced Options for fine-tuning
    with st.expander("Advanced Options", expanded=advanced_expanded):
        options_list = [
            "Base Stable Diffusion V1-5",
            "Dreamshaper-7 (fine-tuned SD V1-5)",
            "Base Stable Diffusion V2-1 (No LCM-LoRA support)"
        ]
        default_model = preset_model if preset_model is not None else "Base Stable Diffusion V1-5"
        default_index = options_list.index(default_model)
        model_option = st.selectbox("Select Model Card", options=options_list, index=default_index)
        
        col_left, col_right = st.columns(2)
        # Left Column: Keyframe Generator Parameters
        with col_left:
            st.markdown("##### Keyframe Generator Parameters")
            num_frames = st.number_input("Number of keyframes (2–200)", min_value=2, max_value=200, value=20)
            if model_option == "Base Stable Diffusion V2-1 (No LCM-LoRA support)":
                enable_lcm_lora = st.checkbox(
                    "Enable LCM-LoRA (accelerated inference, slight decrease in quality)",
                    value=False,
                    disabled=True,
                    help="LCM-LoRA is not available for the selected model card."
                )
            else:
                lcm_default = preset_lcm if preset_lcm is not None else False
                enable_lcm_lora = st.checkbox(
                    "Enable LCM-LoRA (accelerated inference, slight decrease in quality)",
                    value=lcm_default
                )
            use_adain = st.checkbox("Use AdaIN", value=True)
            use_reschedule = st.checkbox("Use reschedule sampling", value=True)
            keyframe_duration = st.number_input("Keyframe Duration (seconds, only if not using FILM)", min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        # Right Column: Inter-frame Interpolator Parameters (FILM)
        with col_right:
            st.markdown("<div class='right-column-divider'>", unsafe_allow_html=True)
            st.markdown("##### Inter-frame Interpolator Parameters")
            default_use_film = preset_film if preset_film is not None else True
            use_film = st.checkbox("Use FILM interpolation", value=default_use_film)
            film_fps = st.number_input("FILM FPS (1–120)", min_value=1, max_value=120, value=30)
            film_recursions = st.number_input("FILM recursion passes (1–6)", min_value=1, max_value=6, value=3)
            st.markdown("</div>", unsafe_allow_html=True)

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

            # Build the CLI command. Note: numeric parameters are converted to strings for CLI compatibility.
            cmd = [
                sys.executable, "run_morphing.py",
                "--model_path", model_option if model_option != "Dreamshaper-7 (fine-tuned SD V1-5)" else "lykon/dreamshaper-7",
                "--image_path_0", imgA_path,
                "--image_path_1", imgB_path,
                "--prompt_0", prompt_A,
                "--prompt_1", prompt_B,
                "--output_path", output_dir,
                "--film_output_folder", film_output_dir,
                "--num_frames", str(num_frames),
                "--keyframe_duration", str(keyframe_duration)
            ]
            if enable_lcm_lora:
                cmd.append("--use_lcm")
            if use_adain:
                cmd.append("--use_adain")
            if use_reschedule:
                cmd.append("--use_reschedule")
            if use_film:
                cmd.append("--use_film")
            cmd.extend(["--film_fps", str(film_fps)])
            cmd.extend(["--film_num_recursions", str(film_recursions)])

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
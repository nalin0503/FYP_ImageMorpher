import os
import sys
import subprocess
import base64
import datetime
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
    """Save an uploaded file to a destination path."""
    with open(dst_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def get_img_as_base64(img):
    """Convert PIL Image to base64 for embedding in HTML."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def ensure_scripts_exist():
    """Check if the required script files exist."""
    required_scripts = ["run_morphing.py", "FILM.py"]
    missing_scripts = [script for script in required_scripts if not os.path.exists(script)]
    
    if missing_scripts:
        error_msg = f"Missing required script(s): {', '.join(missing_scripts)}"
        return False, error_msg
    return True, ""

def create_temp_folder():
    """Create a persistent temporary folder in the repo for processing."""
    base_folder = os.path.join(os.getcwd(), "temp_run")
    os.makedirs(base_folder, exist_ok=True)
    # Create a subfolder with a timestamp to avoid collisions
    run_folder = os.path.join(base_folder, datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_folder)
    return run_folder

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

    # Check if required scripts exist
    scripts_exist, error_msg = ensure_scripts_exist()
    if not scripts_exist:
        st.error(error_msg)
        st.error("Please make sure all required scripts are in the same directory as this Streamlit app.")
        return

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
        except Exception as e:
            st.warning(f"Logo could not be loaded: {e}")

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
            st.image(uploaded_image_A, caption="Preview - Image A", use_container_width=True)
        prompt_A = st.text_input("Prompt for Image A (optional)", value="", key="promptA")
    with col_imgB:
        st.markdown("#### Image B")
        uploaded_image_B = st.file_uploader("Upload your second image", type=["png", "jpg", "jpeg"], key="imgB")
        if uploaded_image_B is not None:
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
            "Maximum quality, longest inference time 🏆",
            "Medium quality, medium inference time ⚖️",
            "Low quality, shortest inference time ⚡",
            "Creative morph 🎨",
            "Custom ⚙️"
        ],
        index=0,
        label_visibility="collapsed"  # Hide the label in the UI but keep it for accessibility
    )

    # Determine preset defaults based on selection
    if preset_option.startswith("Maximum quality"):
        preset_model = "Base Stable Diffusion V2-1"
        preset_film = True
        preset_lcm = False
    elif preset_option.startswith("Medium quality"):
        preset_model = "Base Stable Diffusion V2-1"
        preset_film = False
        preset_lcm = False
    elif preset_option.startswith("Low quality"):
        preset_model = "Base Stable Diffusion V1-5"
        preset_film = False
        preset_lcm = True
    elif preset_option.startswith("Creative morph"):
        preset_model = "Dreamshaper-7 (fine-tuned SD V1-5)"
        preset_film = True
        preset_lcm = True
    else:
        # "Custom"
        preset_model = None
        preset_film = None
        preset_lcm = None

    advanced_expanded = True if preset_option.endswith("⚙️") else False

    # Advanced Options for fine-tuning
    with st.expander("Advanced Options", expanded=advanced_expanded):
        options_list = [
            "Base Stable Diffusion V1-5",
            "Dreamshaper-7 (fine-tuned SD V1-5)",
            "Base Stable Diffusion V2-1"
        ]
        default_model = preset_model if preset_model is not None else "Base Stable Diffusion V1-5"
        default_index = options_list.index(default_model)
        model_option = st.selectbox("Select Model Card", options=options_list, index=default_index)
        
        col_left, col_right = st.columns(2)
        # Left Column: Keyframe Generator Parameters
        with col_left:
            st.markdown("##### Keyframe Generator Parameters")
            num_frames = st.number_input("Number of keyframes (2–50)", min_value=2, max_value=50, value=16)
            lcm_default = preset_lcm if preset_lcm is not None else False
            enable_lcm_lora = st.checkbox(
                "Enable LCM-LoRA",
                value=lcm_default,
                help="Accelerates inference with slight quality decrease"
            )
            use_adain = st.checkbox("Use AdaIN", value=True, help="Adaptive Instance Normalization for improved generation")
            use_reschedule = st.checkbox("Use reschedule sampling", value=True, help="Better sampling strategy")
        
        # Right Column: Inter-frame Interpolator Parameters (FILM)
        with col_right:
            st.markdown("<div class='right-column-divider'>", unsafe_allow_html=True)
            st.markdown("##### Inter-frame Interpolator Parameters")
            default_use_film = preset_film if preset_film is not None else True
            use_film = st.checkbox("Use FILM interpolation", value=default_use_film, help="Frame Interpolation for Large Motion - creates smooth transitions")
            film_recursions = st.number_input("FILM recursion passes (1–6)", min_value=1, max_value=6, value=3, 
                                             help="Higher values create more intermediate frames (smoother but slower)")
            # Set default FPS based on whether FILM is enabled
            default_fps = 30 if use_film else 4
            output_fps = st.number_input("Output FPS (1–120)", min_value=1, max_value=120, value=default_fps, 
                                    help="Output video frames per second")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- SECTION 3: EXECUTE MORPH PIPELINE ----------------
    st.subheader("3. Generate Morphing Video")
    st.markdown("Once satisfied with your inputs, click below to start the process.")

    # New checkbox for SLAB execution toggle
    using_slab = st.checkbox("Using SLAB GPU Cluster?", value=False, help="If enabled, the pipeline command will be prefixed with SLAB cluster execution parameters.")

    if st.button("Run Morphing Pipeline", key="run_pipeline"):
        if not (uploaded_image_A and uploaded_image_B):
            st.error("Please upload both images before running the morphing pipeline.")
            return

        # Instead of using /tmp, create a folder in the repo for temporary processing.
        temp_dir = create_temp_folder()
        
        try:
            # Save uploaded images
            imgA_path = os.path.join(temp_dir, "imageA.png")
            imgB_path = os.path.join(temp_dir, "imageB.png")
            save_uploaded_file(uploaded_image_A, imgA_path)
            save_uploaded_file(uploaded_image_B, imgB_path)
            
            # Create output directories
            output_dir = os.path.join(temp_dir, "morph_results")
            film_output_dir = os.path.join(temp_dir, "film_output")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(film_output_dir, exist_ok=True)
            
            actual_model_path = (
                "lykon/dreamshaper-7" if model_option == "Dreamshaper-7 (fine-tuned SD V1-5)" 
                else "stabilityai/stable-diffusion-2-1-base" if model_option == "Base Stable Diffusion V2-1"
                else "sd-legacy/stable-diffusion-v1-5"
            )
            
            # Build the command for run_morphing.py
            cmd = [
                sys.executable, "run_morphing.py",
                "--model_path", actual_model_path,
                "--image_path_0", imgA_path,
                "--image_path_1", imgB_path,
                "--prompt_0", prompt_A,
                "--prompt_1", prompt_B,
                "--output_path", output_dir,
                "--film_output_folder", film_output_dir,
                "--num_frames", str(num_frames),
                "--fps", str(output_fps)
            ]
            
            if enable_lcm_lora:
                cmd.append("--use_lcm")
            if use_adain:
                cmd.append("--use_adain")
            if use_reschedule:
                cmd.append("--use_reschedule")
            if use_film:
                cmd.append("--use_film")
            
            # Add film recursion parameter
            cmd.extend(["--film_num_recursions", str(film_recursions)])
            
            # If SLAB execution is enabled, prepend the srun command prefix.
            if using_slab:
                slab_prefix = [
                    "srun", "-p", "rtx3090_slab", "-w", "slabgpu05", "--gres=gpu:1",
                    "--job-name=test", "--kill-on-bad-exit=1"
                ]
                cmd = slab_prefix + cmd
            
            st.info("Initializing pipeline. This may take a few minutes...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Update progress status
            for i in range(1, 11):
                status_text.text(f"Step {i}/10: {'Preparing images' if i <= 2 else 'Generating keyframes' if i <= 6 else 'Interpolating frames' if i <= 9 else 'Finalizing video'}")
                progress_bar.progress(i * 10)
                
                if i == 3: # Start processing!
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        st.error(f"Error running morphing pipeline: {e}")
                        return
                    break

            # Set to 100% when done
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            # Check for output video
            video_found = False
            possible_outputs = [f for f in os.listdir(film_output_dir) if f.endswith(".mp4")]
            if possible_outputs:
                final_video_path = os.path.join(film_output_dir, possible_outputs[0])
                video_found = True
            
            if not video_found:
                possible_outputs = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
                if possible_outputs:
                    final_video_path = os.path.join(output_dir, possible_outputs[0])
                    video_found = True
            
            if video_found:
                st.success("Morphing complete! 🎉")
                st.video(final_video_path)
                try:
                    with open(final_video_path, "rb") as f:
                        video_bytes = f.read()
                        st.download_button(
                            "Download Result Video",
                            data=video_bytes,
                            file_name="metamorph_result.mp4",
                            mime="video/mp4"
                        )
                except Exception as e:
                    st.error(f"Error preparing video for download: {e}")
            else:
                st.warning("No output video was generated. Check logs for details.")
                    
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
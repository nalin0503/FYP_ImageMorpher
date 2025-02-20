# Variables
PYTHON = python # or python3
MAIN_SCRIPT = run_morphing.py
OUTPUT_DIR = ./results
FILM_OUTPUT = ./FILM_Results

# Default target: Run the entire morphing pipeline
all: submodule morph

# Ensure the DiffMorpher submodule is initialized and updated
submodule:
	@echo "Initializing and updating submodules..."
	git submodule init
	git submodule sync
	git submodule update --remote
	

# Initialize environment by installing dependencies and (optionally) submodules
init: submodule
	@echo "Installing required Python packages..."
	$(PYTHON) -m pip install -r requirements.txt

# Run morphing pipeline with FILM
# (TODO - configure CLI run command to be able to run make morph FRAMES=30 FILM=true)
morph:
	$(PYTHON) $(MAIN_SCRIPT) \
		--image_path_0 ./assets/Trump.jpg \
		--prompt_0 "A photo of an American man" \
		--image_path_1 ./assets/Biden.jpg \
		--prompt_1 "A photo of an American man" \
		--output_path $(OUTPUT_DIR) \
		--use_adain \
		--use_reschedule \
		--save_inter \
		--num_frames 16 \
		--duration 100 \
		--use_film \
		--film_fps 30 \
		--film_num_recursions 3

### Add more options here later, make install, make init (install reqs)

# Help message
help:
	@echo "Makefile for Image Morphing Project"
	@echo "Available commands:"
	@echo "  make           - Initialize submodules and run the entire morphing pipeline"
	@echo "  make init      - Install dependencies and set up the environment"
	@echo "  make submodule - Initialize and update Git submodules"
	@echo "  make morph     - Run the morphing pipeline with FILM"
	@echo "  make help      - Show this help message"

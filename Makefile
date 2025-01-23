# Rough Makefile (W.I.P.)

# Variables
PYTHON = python # or python3
MAIN_SCRIPT = run_morphing.py
OUTPUT_DIR = ./results
FILM_OUTPUT = ./FILM_Results

# Default target: Run the entire morphing pipeline
all: morph

# Run morphing pipeline with FILM, TODO - configure to be able to run make morph FRAMES=30 FILM=true
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

# Clean output directories
clean:
	rm -rf $(OUTPUT_DIR)/*
	rm -rf $(FILM_OUTPUT)/*

# Help message
help:
	@echo "Makefile for Image Morphing Project"
	@echo "Available commands:"
	@echo "  make           - Run the entire morphing pipeline"
	@echo "  make morph     - Same as 'make'; generates morphing video with FILM"
	@echo "  make clean     - Clean up all output directories"
	@echo "  make help      - Show this help message"

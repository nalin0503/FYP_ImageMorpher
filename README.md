# Image Morphing Project 
NTU Final Year Project on Image Morphing via Consistency / Diffusion Models enhanced by FILM.

## Sample run commands: 
`run_morphing.py` is the overall orchestration script here. 

### **Option 1: Direct CLI Command**
```bash
python run_morphing.py \
  --image_path_0 ./assets/Trump.jpg \
  --prompt_0 "A photo of an American man" \
  --image_path_1 ./assets/Biden.jpg \
  --prompt_1 "A photo of an American man" \
  --output_path ./results/Trump_Biden \
  --use_adain \
  --use_reschedule \
  --save_inter \
  --num_frames 16 \
  --duration 100 \
  --use_film \
  --film_fps 30 \
  --film_num_recursions 3
```

### **Option 2: Using Makefile**
```bash
make morph
```
---
---

### My Dev Notes (W.I.P.)

Technical impl stuff:
1) pytorch 2.0 for acceleration.... (explore, see email draft)
2) text embedding interpolation.. slerp?  
3) Test on diffmorpher eval dataset, test for smoothness basically, or image fidelity between new and old to measure quality (quantitative, use FID or sth) - see ipad for sched!! 
4) add google style PEP type-hinting 
5) add requirements.txt, or a poetry, or docker/conda image... FOR SETUP use makefile!! 


Some ideas: 
1) LLM support for dummies ? 
2) adain normalisation and image correction as a post-processing step... 
3) to tune the right number of steps... check with FID and pick the lowest one 
4) use auto image captioning for better initialisation of target images' prompts (can do manually too.. but to remove this req), 
5) then generate intermediates using text prompt interpolation (below)
6)  weighting the linear combi seems interesting (so weight the lora params and then addd... like an alpha for that (Check lcm-lora image morphing))
7) Colour correction methods on the final morph video to make it seem smoother, higher fidelity. 

explore aidi instead of DDIM for inversion? no.

Later:
- text embed interpolate 
- And obv FILM (in progress)
- post on HF as a website (More like HF demo....)
- need to make a seperate webapp for the functionality you want... (later, finished product, see FYP pres/report notes)
- Back compute from number of frames (pre set, drop down) to the k - better usability for user


# Image Morphing Project 
NTU Final Year Project on Image Morphing via Consistency / Diffusion Models

Technical impl stuff:
1) pytorch 2.0 for acceleration.... (explore, see email draft)
2) text embedding interpolation.. slerp?  
3) Test on diffmorpher eval dataset, test for smoothness basically, or image fidelity between new and old to measure quality (quantitative, use FID or sth)


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

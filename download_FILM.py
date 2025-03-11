import kagglehub

# Download latest version
path = kagglehub.model_download("google/film/tensorFlow2/film")

print("Path to model files:", path)
from safetensors.torch import safe_open, save_file
import torch

# Path to your original safetensors file
safetensors_path = "/fs/nexus-projects/brain_project/microsoft/cache/hub/models--stabilityai--stable-audio-open-1.0/snapshots/4ab2b18994346363f65d4acbe7b034e814d99040/model.safetensors"
# Path to the new safetensors file
new_safetensors_path = "sonal.safetensors"
# String to match keys
match_string = "pretransform"  # Replace with the string you want to match

# Load the original safetensors file
with safe_open(safetensors_path, framework="pt") as f:
    state_dict = {}
    for key in f.keys():
        if match_string in key:
            state_dict[key] = f.get_tensor(key)

# Save the filtered state_dict to a new safetensors file
save_file(state_dict, new_safetensors_path)

print(f"Filtered model saved to {new_safetensors_path}")
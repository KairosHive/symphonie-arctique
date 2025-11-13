import json
import urllib.request
import urllib.error
import os
import time
import uuid
import random

# --- Configuration ---
# The address of your ComfyUI server
COMFYUI_ADDRESS = "127.0.0.1:8188"
# The JSON file containing the prompts
PROMPTS_FILE_PATH = "./assets/song_joey__timeline_joey__chunk1p0s__top5_prompts-macro.json"
# The JSON file containing the ComfyUI API workflow
API_WORKFLOW_FILE_PATH = "./assets/Flux-dev_api.json"
# The output file for image metadata
METADATA_FILE_PATH = "./output/image_metadata.json"
# A unique ID for this client
CLIENT_ID = str(uuid.uuid4())

# --- Helper Functions ---

def load_json_file(filepath):
    """Loads a JSON file from the given filepath."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{filepath}' is not a valid JSON file.")
        return None

def queue_prompt(prompt_workflow):
    """Queues a prompt workflow to the ComfyUI server using urllib."""
    payload = {"prompt": prompt_workflow, "client_id": CLIENT_ID}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"http://{COMFYUI_ADDRESS}/prompt", data=data)
    
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(f"Error communicating with ComfyUI server: HTTP {e.code} {e.reason}")
        print(f"  Response: {e.read().decode('utf-8')}")
        return None
    except urllib.error.URLError as e:
        print(f"Error communicating with ComfyUI server: {e.reason}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- NEW: Function to save metadata ---
def save_metadata(filepath, key, data):
    """Loads, updates, and saves metadata to a JSON file in a robust way."""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Load existing data from the file, or create an empty dict if it doesn't exist/is invalid
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}
        
    # Update the dictionary with the new data
    metadata[key] = data
    
    # Write the updated dictionary back to the file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
    except IOError as e:
        print(f"Error: Could not write to metadata file '{filepath}'. Reason: {e}")


# --- Main Script Logic ---

def main():
    """Main function to load data and run the prompt queue."""
    print("--- Starting ComfyUI Prompt Runner ---")

    # 1. Load the JSON files
    prompts_data = load_json_file(PROMPTS_FILE_PATH)
    api_workflow = load_json_file(API_WORKFLOW_FILE_PATH)

    if not prompts_data or not api_workflow:
        print("Aborting due to file loading errors.")
        return

    # Node IDs to be modified
    CLIP_TEXT_ENCODE_NODE_ID = "5"
    SAVE_IMAGE_NODE_ID = "23"
    SAMPLER_NODE_ID = "3"
    LATENT_NODE_ID = "6"

    # 2. Iterate through each section and each prompt
    total_prompts = sum(len(section.get('generated_prompts', [])) for section in prompts_data.get('sections', []))
    current_prompt_num = 0

    for section in prompts_data.get('sections', []):
        section_name = section.get('name', 'untitled_section')
        generated_prompts = section.get('generated_prompts', [])
        
        if not generated_prompts:
            continue

        for index, prompt_data in enumerate(generated_prompts):
            prompt_text = prompt_data.get('prompt')
            
            if not prompt_text:
                print(f"Warning: Skipping malformed prompt object at index {index} in section '{section_name}'.")
                continue

            current_prompt_num += 1
            print(f"\n({current_prompt_num}/{total_prompts}) Processing: Section '{section_name}', Prompt {index + 1}")
            print(f"  > Prompt: {prompt_text[:100]}...")

            # 3. Create a fresh copy of the workflow for modification
            current_workflow = json.loads(json.dumps(api_workflow))

            # --- CHANGED: Define filename prefix and seed before modifying the workflow ---
            filename_prefix = f"FontedesGlaces/{section_name}_{index + 1:02d}"
            noise_seed = random.randint(0, 2**64 - 1)

            # 4. Modify the workflow with the current prompt
            if CLIP_TEXT_ENCODE_NODE_ID in current_workflow:
                current_workflow[CLIP_TEXT_ENCODE_NODE_ID]["inputs"]["clip_l"] = prompt_text
                current_workflow[CLIP_TEXT_ENCODE_NODE_ID]["inputs"]["t5xxl"] = prompt_text
            else:
                print(f"Warning: Node ID '{CLIP_TEXT_ENCODE_NODE_ID}' not found in the workflow.")
            
            if LATENT_NODE_ID in current_workflow:
                current_workflow[LATENT_NODE_ID]["inputs"]["width"] = 1280
                current_workflow[LATENT_NODE_ID]["inputs"]["height"] = 720
                current_workflow[LATENT_NODE_ID]["inputs"]["batch_size"] = 3
                
            if SAVE_IMAGE_NODE_ID in current_workflow:
                current_workflow[SAVE_IMAGE_NODE_ID]["inputs"]["filename_prefix"] = filename_prefix
            else:
                print(f"Warning: Node ID '{SAVE_IMAGE_NODE_ID}' not found in the workflow.")

            if SAMPLER_NODE_ID in current_workflow:
                current_workflow[SAMPLER_NODE_ID]["inputs"]["noise_seed"] = noise_seed
            else:
                print(f"Warning: Node ID '{SAMPLER_NODE_ID}' not found in the workflow.")

            # --- NEW: Save metadata for the current prompt and seed ---
            print(f"  > Saving metadata. Seed: {noise_seed}")
            image_metadata = {
                "prompt": prompt_text,
                "seed": noise_seed
            }
            save_metadata(METADATA_FILE_PATH, filename_prefix, image_metadata)

            # 5. Queue the prompt
            print("  > Submitting to ComfyUI...")
            response = queue_prompt(current_workflow)
            if response:
                prompt_id = response.get('prompt_id', 'N/A')
                print(f"  > Successfully queued prompt. Prompt ID: {prompt_id}")
            else:
                print("  > Failed to queue prompt.")
            
            time.sleep(1)

    print("\n--- All prompts have been queued. ---")

if __name__ == "__main__":
    main()
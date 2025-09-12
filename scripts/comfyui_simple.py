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
PROMPTS_FILE_PATH = "./assets/timeline_joey_prompts.json"
# The JSON file containing the ComfyUI API workflow
API_WORKFLOW_FILE_PATH = "./assets/Flux-dev_textures_api.json"
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

    # Node IDs to be modified (as per the provided JSON files)
    # 5: The main text prompt node (CLIPTextEncodeFlux)
    # 23: The save image node (SaveImage)
    # 3: The sampler node (XlabsSampler) for randomizing the seed
    CLIP_TEXT_ENCODE_NODE_ID = "5"
    SAVE_IMAGE_NODE_ID = "23"
    SAMPLER_NODE_ID = "3"
    LATENT_NODE_ID = "6"

    # 2. Iterate through each section and each prompt
    total_prompts = sum(len(section.get('prompts', [])) for section in prompts_data.get('sections', []))
    current_prompt_num = 0

    for section in prompts_data.get('sections', []):
        section_name = section.get('name', 'untitled_section')
        prompts = section.get('prompts', [])
        
        if not prompts:
            continue

        for index, prompt_text in enumerate(prompts):
            current_prompt_num += 1
            print(f"\n({current_prompt_num}/{total_prompts}) Processing: Section '{section_name}', Prompt {index + 1}")
            print(f"  > Prompt: {prompt_text[:100]}...") # Print first 100 chars

            # 3. Create a fresh copy of the workflow for modification
            current_workflow = json.loads(json.dumps(api_workflow))

            # 4. Modify the workflow with the current prompt
            # Update the positive prompt text in the CLIPTextEncodeFlux node
            if CLIP_TEXT_ENCODE_NODE_ID in current_workflow:
                current_workflow[CLIP_TEXT_ENCODE_NODE_ID]["inputs"]["clip_l"] = prompt_text
                current_workflow[CLIP_TEXT_ENCODE_NODE_ID]["inputs"]["t5xxl"] = prompt_text
            else:
                print(f"Warning: Node ID '{CLIP_TEXT_ENCODE_NODE_ID}' not found in the workflow.")
            if LATENT_NODE_ID in current_workflow:
                current_workflow[LATENT_NODE_ID]["inputs"]["width"] = 1280
                current_workflow[LATENT_NODE_ID]["inputs"]["height"] = 720

            # Update the output filename in the SaveImage node
            if SAVE_IMAGE_NODE_ID in current_workflow:
                filename_prefix = f"Flux/{section_name}_{index + 1:02d}"
                current_workflow[SAVE_IMAGE_NODE_ID]["inputs"]["filename_prefix"] = filename_prefix
            else:
                 print(f"Warning: Node ID '{SAVE_IMAGE_NODE_ID}' not found in the workflow.")

            # Update the seed in the sampler to a random value for variety
            if SAMPLER_NODE_ID in current_workflow:
                current_workflow[SAMPLER_NODE_ID]["inputs"]["noise_seed"] = random.randint(0, 2**64 - 1)
            else:
                 print(f"Warning: Node ID '{SAMPLER_NODE_ID}' not found in the workflow.")


            # 5. Queue the prompt
            print("  > Submitting to ComfyUI...")
            response = queue_prompt(current_workflow)
            if response:
                prompt_id = response.get('prompt_id', 'N/A')
                print(f"  > Successfully queued prompt. Prompt ID: {prompt_id}")
            else:
                print("  > Failed to queue prompt.")
            
            # Optional: Add a small delay between requests to not overwhelm the server
            time.sleep(1)

    print("\n--- All prompts have been queued. ---")

if __name__ == "__main__":
    main()


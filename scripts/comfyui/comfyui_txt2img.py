import json
import urllib.request
import urllib.error
import os
import time
import uuid
import random
import argparse

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

def queue_prompt(prompt_workflow, client_id, server_address):
    """Queues a prompt workflow to the ComfyUI server using urllib."""
    payload = {"prompt": prompt_workflow, "client_id": client_id}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    
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

def save_metadata(filepath, key, data):
    """Loads, updates, and saves metadata to a JSON file in a robust way."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}
        
    metadata[key] = data
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
    except IOError as e:
        print(f"Error: Could not write to metadata file '{filepath}'. Reason: {e}")

def create_model_hash(workflow):
    """Creates a hash based on model-related nodes to help with caching."""
    model_nodes = {}
    # Include all model loading nodes with consistent IDs
    for node_id in ["4", "8", "49", "57", "64", "65"]:  # Your model loading nodes
        if node_id in workflow:
            model_nodes[node_id] = workflow[node_id]
    return hash(json.dumps(model_nodes, sort_keys=True))

def main():
    """Main function with model caching optimization."""
    parser = argparse.ArgumentParser(description="Queue ComfyUI image generation prompts and save metadata.")
    parser.add_argument("--server", type=str, default="127.0.0.1:8188", help="Address of the ComfyUI server.")
    parser.add_argument("--prompts", type=str, default="./results/song_simon_cut__timeline_simon__chunk1p0s__top5_prompts-creatures.json", help="Path to the prompts JSON file.")
    parser.add_argument("--workflow", type=str, default="./assets/txt2img_flux-dev-lorastack_api.json", help="Path to the ComfyUI API workflow JSON file.")
    parser.add_argument("--metadata", type=str, default="./output/image_metadata.json", help="Path to the output metadata JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Output folder for generated images.")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=720, help="Height of the generated image.")
    parser.add_argument("--batch", type=int, default=3, help="Number of images to generate per prompt.")
    args = parser.parse_args()

    client_id = str(uuid.uuid4())
    print("--- Starting ComfyUI Prompt Runner with Model Caching ---")
    print(f"Settings: {args.width}x{args.height}, batch size: {args.batch}")

    prompts_data = load_json_file(args.prompts)
    api_workflow = load_json_file(args.workflow)

    if not prompts_data or not api_workflow:
        print("Aborting due to file loading errors.")
        return
    
    # if args.metadata already exists, add number suffix to avoid overwriting
    base_metadata_path = args.metadata
    counter = 1
    while os.path.exists(args.metadata):
        name, ext = os.path.splitext(base_metadata_path)
        args.metadata = f"{name}_{counter:02d}{ext}"
        counter += 1
    if counter > 1:
        print(f"  > Metadata file exists. Using new file: {args.metadata}")
    
    save_metadata(args.metadata, "settings", { 
        "width": args.width,
        "height": args.height,
        "batch_size": args.batch
    })
    
    # Node IDs
    CLIP_TEXT_ENCODE_NODE_ID = "5"
    SAVE_IMAGE_NODE_ID = "23"
    SAMPLER_NODE_ID = "3"
    LATENT_NODE_ID = "6"

    # Create a base model hash for consistency
    base_model_hash = create_model_hash(api_workflow)
    print(f"Base model hash: {base_model_hash}")

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

            # Create workflow copy
            current_workflow = json.loads(json.dumps(api_workflow))

            filename_prefix = f"{args.output}/{section_name}/{section_name}_{index + 1:02d}"
            noise_seed = random.randint(0, 2**64 - 1)

            # IMPORTANT: Only modify the nodes that need to change
            # Don't modify model loading nodes unnecessarily
            if CLIP_TEXT_ENCODE_NODE_ID in current_workflow:
                current_workflow[CLIP_TEXT_ENCODE_NODE_ID]["inputs"]["clip_l"] = prompt_text
                current_workflow[CLIP_TEXT_ENCODE_NODE_ID]["inputs"]["t5xxl"] = prompt_text

            if LATENT_NODE_ID in current_workflow:
                current_workflow[LATENT_NODE_ID]["inputs"]["width"] = args.width
                current_workflow[LATENT_NODE_ID]["inputs"]["height"] = args.height
                current_workflow[LATENT_NODE_ID]["inputs"]["batch_size"] = args.batch
                
            if SAVE_IMAGE_NODE_ID in current_workflow:
                current_workflow[SAVE_IMAGE_NODE_ID]["inputs"]["filename_prefix"] = filename_prefix

            if SAMPLER_NODE_ID in current_workflow:
                current_workflow[SAMPLER_NODE_ID]["inputs"]["noise_seed"] = noise_seed

            # Save metadata
            print(f"  > Saving metadata. Seed: {noise_seed}")
            image_metadata = {
                "prompt": prompt_text,
                "seed": noise_seed,
                "width": args.width,
                "height": args.height,
                "batch_size": args.batch
            }
            save_metadata(args.metadata, filename_prefix, image_metadata)

            # Queue the prompt
            print("  > Submitting to ComfyUI...")
            response = queue_prompt(current_workflow, client_id, args.server)
            if response:
                prompt_id = response.get('prompt_id', 'N/A')
                print(f"  > Successfully queued prompt. Prompt ID: {prompt_id}")
            else:
                print("  > Failed to queue prompt.")
            
            # Shorter delay between prompts to maintain model in memory
            time.sleep(0.5)

    print("\n--- All prompts have been queued. ---")

if __name__ == "__main__":
    main()
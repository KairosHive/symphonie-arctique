import json
import urllib.request
import urllib.error
import os
import time
import uuid
import random
import argparse

# --- Helper Functions ---

def load_jsonl_file(filepath):
    """Loads a JSONL file, parsing each line as a separate JSON object."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {filepath}: {line.strip()}")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None

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
    """Queues a prompt workflow to the ComfyUI server."""
    payload = {"prompt": prompt_workflow, "client_id": client_id}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"An error occurred while queueing the prompt: {e}")
        return None

def save_metadata(filepath, key, data):
    """Loads, updates, and saves metadata to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}
    metadata[key] = data
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

# --- FIX: Rewritten for better reliability ---
def clean_motion_prompt(text: str) -> str:
    """
    Cleans the motion prompt by first attempting to parse it as JSON,
    and falling back to manual string cleaning if that fails.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.strip()

    # First, try to parse the whole string as a JSON object
    try:
        data = json.loads(text)
        if isinstance(data, dict) and 'motion_prompt' in data:
            # If it's a JSON object with the key, we found the prompt. No more cleaning needed.
            return str(data['motion_prompt']).strip()
    except (json.JSONDecodeError, TypeError):
        # It's not a clean JSON string, so we fall back to manual cleaning.
        pass

    # Manual cleaning for strings with trailing junk.
    # We find the last meaningful punctuation and truncate after it.
    end_chars = ['”', '"', '.']
    last_pos = -1
    for char in end_chars:
        # rfind finds the last occurrence of the character
        pos = text.rfind(char)
        if pos > last_pos:
            last_pos = pos

    if last_pos != -1:
        text = text[:last_pos + 1]

    # Final cleanup of any leading/trailing quotes or whitespace
    return text.strip().strip('"“”')


# --- Main Script Logic ---

def main():
    parser = argparse.ArgumentParser(description="Queue ComfyUI image-to-video generation prompts.")
    parser.add_argument("--server", type=str, default="127.0.0.1:8188", help="Address of the ComfyUI server.")
    parser.add_argument("--prompts", type=str, required=True, help="Path to the motion prompts JSONL file.")
    parser.add_argument("--workflow", type=str, required=True, help="Path to the ComfyUI I2V API workflow.")
    parser.add_argument("--footage_dir", type=str, required=True, help="Absolute path to the 'footage' directory on the ComfyUI server's filesystem.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated videos. Can be a single string")
    parser.add_argument("--metadata", type=str, default="./output/i2v_metadata.json", help="Path for the output metadata file.")
    parser.add_argument("--frames", type=int, default=25, help="Number of frames to generate for each video.")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the output video.")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video.")
    parser.add_argument("--height", type=int, default=720, help="Height of the generated video.")
    args = parser.parse_args()

    client_id = str(uuid.uuid4())
    print("--- Starting ComfyUI Image-to-Video Prompt Runner ---")
    print(f"Settings: {args.frames} frames @ {args.fps}fps ({args.width}x{args.height})")

    # 1. Load the prompt and workflow files
    prompts_data = load_jsonl_file(args.prompts)
    api_workflow = load_json_file(args.workflow)
    output_dir = args.output_dir

    if not prompts_data or not api_workflow:
        print("Aborting due to file loading errors.")
        return

    # --- Node IDs for the Wan2.2 I2V workflow (video_wan2_2_14B_i2v.json) ---
    # These IDs must match the workflow JSON. Update if workflow changes.
    POSITIVE_PROMPT_NODE_ID = "6"   # CLIPTextEncode (Positive Prompt)
    NEGATIVE_PROMPT_NODE_ID = "7"   # CLIPTextEncode (Negative Prompt)
    LOAD_IMAGE_NODE_ID = "62"       # LoadImage - source image for I2V
    I2V_NODE_ID = "63"              # WanImageToVideo - controls width, height, length
    SAVE_VIDEO_NODE_ID = "61"       # SaveVideo
    CREATE_VIDEO_NODE_ID = "60"     # CreateVideo - controls FPS
    SAMPLER_NODE_ID_1 = "57"        # KSamplerAdvanced (high noise, first pass)
    SAMPLER_NODE_ID_2 = "58"        # KSamplerAdvanced (low noise, second pass)

    # 2. Iterate through each prompt in the JSONL file
    total_prompts = len(prompts_data)
    # randomize order of (index of) prompts to get a better variety if interrupted
    prompt_indices = list(range(total_prompts))
    random.shuffle(prompt_indices)

    for i in prompt_indices:
        prompt_data = prompts_data[i]
        motion_prompt = prompt_data.get('motion_prompt')
        relative_image_path = prompt_data.get('image_rel')
        image_name = prompt_data.get('image_name', f'unknown_image_{i}')

        if not motion_prompt or not relative_image_path:
            print(f"Warning: Skipping item {i+1} due to missing 'motion_prompt' or 'image_rel'.")
            continue
        
        # Clean the motion prompt text
        motion_prompt = clean_motion_prompt(motion_prompt)

        print(f"\n({i+1}/{total_prompts}) Processing Image: {image_name}")
        print(f"  > Motion Prompt: {motion_prompt[:100]}...")
        # This replaces backslashes with forward slashes for cross-platform compatibility. "/")
        # The path must be understandable by the server, not the client running this script.
        server_image_path = os.path.join(args.footage_dir, relative_image_path)
        print(f"  > Source Image Path on Server: {server_image_path}")

        # Use a deep copy of the workflow for each iteration
        current_workflow = json.loads(json.dumps(api_workflow))
        filename_prefix = f"FontedesGlaces/i2v/{output_dir}/{os.path.splitext(image_name)[0]}"
        noise_seed = random.randint(0, 2**64 - 1)

        # 3. Modify the workflow with the current data
        # --- Set motion prompt in the positive prompt node ---
        if POSITIVE_PROMPT_NODE_ID in current_workflow:
            current_workflow[POSITIVE_PROMPT_NODE_ID]["inputs"]["text"] = motion_prompt
        else:
            print(f"Warning: Node {POSITIVE_PROMPT_NODE_ID} (Positive Prompt) not found in workflow.")

        # --- Load source image ---
        if LOAD_IMAGE_NODE_ID in current_workflow:
            current_workflow[LOAD_IMAGE_NODE_ID]["inputs"]["image"] = server_image_path
        else:
            print(f"Warning: Node {LOAD_IMAGE_NODE_ID} (LoadImage) not found in workflow. Cannot load image.")
            continue  # Skip if we can't load the source image

        # --- Set dimensions and length in WanImageToVideo node ---
        if I2V_NODE_ID in current_workflow:
            current_workflow[I2V_NODE_ID]["inputs"]["width"] = args.width
            current_workflow[I2V_NODE_ID]["inputs"]["height"] = args.height
            current_workflow[I2V_NODE_ID]["inputs"]["length"] = args.frames
        else:
            print(f"Warning: Node {I2V_NODE_ID} (WanImageToVideo) not found. Cannot set dimensions/length.")
        
        if SAVE_VIDEO_NODE_ID in current_workflow:
            current_workflow[SAVE_VIDEO_NODE_ID]["inputs"]["filename_prefix"] = filename_prefix
        else:
            print(f"Warning: Node {SAVE_VIDEO_NODE_ID} (SaveVideo) not found.")
        
        if CREATE_VIDEO_NODE_ID in current_workflow:
            current_workflow[CREATE_VIDEO_NODE_ID]["inputs"]["fps"] = args.fps
        else:
            print(f"Warning: Node {CREATE_VIDEO_NODE_ID} (CreateVideo) not found. Cannot set FPS.")
        
        # Set a random seed for both samplers for variety
        if SAMPLER_NODE_ID_1 in current_workflow:
            current_workflow[SAMPLER_NODE_ID_1]["inputs"]["noise_seed"] = noise_seed
        else:
            print(f"Warning: Node {SAMPLER_NODE_ID_1} (KSamplerAdvanced #1) not found.")
        if SAMPLER_NODE_ID_2 in current_workflow:
            # The second sampler's seed is often ignored if "add_noise" is disabled, but we set it for consistency.
            current_workflow[SAMPLER_NODE_ID_2]["inputs"]["noise_seed"] = noise_seed
        else:
            print(f"Warning: Node {SAMPLER_NODE_ID_2} (KSamplerAdvanced #2) not found.")


        # 4. Save metadata and queue the prompt
        print(f"  > Saving metadata. Seed: {noise_seed}, Frames: {args.frames}, FPS: {args.fps}")
        video_metadata = {
            "source_image": server_image_path,
            "motion_prompt": motion_prompt,
            "seed": noise_seed,
            "frames": args.frames,
            "fps": args.fps,
            "width": args.width,
            "height": args.height
        }
        save_metadata(args.metadata, filename_prefix, video_metadata)

        print("  > Submitting to ComfyUI...")
        response = queue_prompt(current_workflow, client_id, args.server)
        if response:
            print(f"  > Successfully queued prompt. Prompt ID: {response.get('prompt_id', 'N/A')}")
        else:
            print("  > Failed to queue prompt.")

        time.sleep(1)

    print("\n--- All prompts have been queued. ---")

if __name__ == "__main__":
    main()
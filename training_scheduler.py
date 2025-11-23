# Code to run training on a server with 3 V100 GPUs, courtesy of Gemini

import subprocess
import os
import time
import sys

# --- CONFIGURATION ---
# The list of GPUs you have available (V100s)
AVAILABLE_GPUS = [0, 1, 2] 

# The exact commands extracted from your model_train_commands.txt
COMMANDS = [
    # 1. Google ViT, CIFAR-100, BN OFF
    "python train_vit_linear.py --dataset cifar100 --model_id google/vit-huge-patch14-224-in21k --model_type vit --batch_size 256 --epochs 25 --precision amp --seed 0",
    
    # 2. Google ViT, CIFAR-100, BN ON
    "python train_vit_linear.py --dataset cifar100 --model_id google/vit-huge-patch14-224-in21k --model_type vit --use_bn_head --batch_size 256 --epochs 25 --precision amp --seed 0",
    
    # 3. Google ViT, ImageNet-100, BN OFF
    "python train_vit_linear.py --dataset imagenet100 --model_id google/vit-huge-patch14-224-in21k --model_type vit --batch_size 256 --epochs 25 --precision amp --seed 0",
    
    # 4. Google ViT, ImageNet-100, BN ON
    "python train_vit_linear.py --dataset imagenet100 --model_id google/vit-huge-patch14-224-in21k --model_type vit --use_bn_head --batch_size 256 --epochs 25 --precision amp --seed 0",
    
    # 5. I-JEPA, CIFAR-100, no last-4
    "python train_vit_linear.py --dataset cifar100 --model_id facebook/ijepa_vith14_22k --model_type ijepa --batch_size 256 --epochs 25 --precision amp --seed 0",
    
    # 6. I-JEPA, CIFAR-100, last-4 concatenated
    "python train_vit_linear.py --dataset cifar100 --model_id facebook/ijepa_vith14_22k --model_type ijepa --last4 --batch_size 256 --epochs 25 --precision amp --seed 0",
    
    # 7. I-JEPA, ImageNet-100, no last-4
    "python train_vit_linear.py --dataset imagenet100 --model_id facebook/ijepa_vith14_22k --model_type ijepa --batch_size 256 --epochs 25 --precision amp --seed 0",
    
    # 8. I-JEPA, ImageNet-100, last-4 concatenated
    "python train_vit_linear.py --dataset imagenet100 --model_id facebook/ijepa_vith14_22k --model_type ijepa --last4 --batch_size 256 --epochs 25 --precision amp --seed 0"
]

def main():
    # Queue of commands to run
    cmd_queue = COMMANDS[:]
    
    # List of currently running processes: [(subprocess_handle, gpu_id), ...]
    running_procs = []
    
    # Pool of currently free GPUs
    free_gpus = AVAILABLE_GPUS[:]

    print(f"--- Starting Scheduler ---")
    print(f"Total Jobs: {len(cmd_queue)}")
    print(f"Available GPUs: {free_gpus}")
    print("--------------------------\n")

    while len(cmd_queue) > 0 or len(running_procs) > 0:
        
        # 1. Check for finished processes
        # We iterate over a copy ([:]) so we can remove items safely
        for p, gpu in running_procs[:]:
            poll = p.poll()
            if poll is not None:
                # Process has finished
                print(f"✅ Job finished on GPU {gpu} (Exit code: {poll})")
                running_procs.remove((p, gpu))
                free_gpus.append(gpu)
        
        # 2. Assign new jobs if we have free GPUs and jobs waiting
        while len(free_gpus) > 0 and len(cmd_queue) > 0:
            gpu = free_gpus.pop(0)
            cmd = cmd_queue.pop(0)
            
            # Create environment specific to this process
            env = os.environ.copy()
            # This forces the script to see ONLY this specific GPU as "cuda:0"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            
            print(f"🚀 Starting job on GPU {gpu}:")
            print(f"   Cmd: {cmd}")
            
            # Start the process in the background
            p = subprocess.Popen(cmd.split(), env=env)
            running_procs.append((p, gpu))

        # 3. Wait a bit before checking again to save CPU cycles
        time.sleep(10)

    print("\n🎉 All jobs completed.")

if __name__ == "__main__":
    main()
import numpy as np
import subprocess
import os
import glob
import sys
import argparse
import cv2
import json

def load_grid_dataset(folder_path, grid_u, grid_v):
    search_path = os.path.join(folder_path, '*.png')
    file_paths = glob.glob(search_path)
    file_paths.sort()
    
    total_expected = grid_u * grid_v
    if len(file_paths) < total_expected:
         print(f"ATTENZIONE: Trovati {len(file_paths)} file, attesi {total_expected}.")
    
    first = cv2.imread(file_paths[0], cv2.IMREAD_COLOR)
    H, W, C = first.shape
    print(f"Dataset: {grid_u}x{grid_v} viste. Risoluzione nativa: {W}x{H}")

    lf_data = np.zeros((grid_u, grid_v, H, W, C), dtype=np.uint8)
    original_names = []

    print("Caricamento immagini...")
    for idx, path in enumerate(file_paths):
        if idx >= total_expected: break
        basename = os.path.basename(path)
        original_names.append(basename)
        
        u = idx % grid_u
        v = idx // grid_u
        
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img.shape != (H, W, C): continue 
        lf_data[u, v, :, :, :] = img

    return lf_data, original_names

def transform_and_pad_epi_safe(lf_data):
    """
    Trasforma in EPI e applica PADDING SICURO per HEVC.
    Garantisce Altezza >= 64 e multipla di 8.
    """
    U, V, H, W, C = lf_data.shape
    
    print("Generazione volume EPI...")
    
    # Trasformazione standard (V, H, U, W, C)
    epi_step1 = lf_data.transpose(1, 2, 0, 3, 4)
    epi_volume = epi_step1.reshape(V * H, U, W, C)
    
    # --- LOGICA PADDING SICURO ---
    current_h = U
    MIN_HEIGHT = 64  # Altezza minima sicura per HEVC/AV1
    ALIGN_TO = 8     # Allineamento blocchi
    
    target_h = current_h
    
    # 1. Deve essere almeno MIN_HEIGHT
    if target_h < MIN_HEIGHT:
        target_h = MIN_HEIGHT
        
    # 2. Deve essere multiplo di ALIGN_TO
    if target_h % ALIGN_TO != 0:
        target_h += (ALIGN_TO - (target_h % ALIGN_TO))
        
    pad_h = target_h - current_h
    
    if pad_h > 0:
        print(f"PADDING DI SICUREZZA: Altezza {current_h} -> {target_h} (+{pad_h} pixel neri)")
        # Applica padding solo in basso sull'asse 1 (altezza)
        epi_volume = np.pad(epi_volume, ((0,0), (0, pad_h), (0,0), (0,0)), mode='constant')
        
    return epi_volume, pad_h

def compress_epi_ffmpeg(epi_volume, output_file, codec='av1'):
    frames, height, width, channels = epi_volume.shape
    
    print(f"--- Compressione ({codec.upper()}) ---")
    print(f"Geometria Video Finale: {width}x{height}, Frames: {frames}")

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', '25',
        '-i', '-', 
    ]
    
    if codec == 'av1':
        cmd.extend(['-c:v', 'libaom-av1', '-lossless', '1', '-cpu-used', '4', '-pix_fmt', 'gbrp'])
    elif codec == 'hevc':
        cmd.extend(['-c:v', 'libx265', '-x265-params', 'lossless=1', '-pix_fmt', 'gbrp'])
    
    cmd.append(output_file)
    
    # stderr=None per vedere errori
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=None, stdout=None)
    
    try:
        chunk_size = 100
        total = len(epi_volume)
        for i in range(0, total, chunk_size):
            chunk = epi_volume[i:i+chunk_size]
            proc.stdin.write(chunk.tobytes())
            
        proc.stdin.close()
    except Exception as e:
        print(f"Errore invio dati: {e}")
        proc.kill()
    
    proc.wait()
    if proc.returncode != 0:
        print("Errore critico FFmpeg.")
        sys.exit(1)

def save_metadata(output_file, grid_u, grid_v, H, W, names, padding):
    meta_file = output_file + ".json"
    data = {
        "grid_u": grid_u,
        "grid_v": grid_v,
        "orig_h": H,
        "orig_w": W,
        "padding_h": int(padding), # Converti numpy int a python int per JSON
        "filenames": names
    }
    with open(meta_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Metadati salvati: {meta_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--u", type=int, required=True)
    parser.add_argument("--v", type=int, required=True)
    parser.add_argument("-o", "--output", default="epi_output.mkv")
    parser.add_argument("-c", "--codec", default="av1", choices=["av1", "hevc"])
    
    args = parser.parse_args()
    
    lf_5d, names = load_grid_dataset(args.input_folder, args.u, args.v)
    H_orig, W_orig = lf_5d.shape[2], lf_5d.shape[3]
    
    epi_vol, pad_val = transform_and_pad_epi_safe(lf_5d)
    
    compress_epi_ffmpeg(epi_vol, args.output, args.codec)
    
    save_metadata(args.output, args.u, args.v, H_orig, W_orig, names, pad_val)
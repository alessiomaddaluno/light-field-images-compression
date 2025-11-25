import numpy as np
import cv2
import os
import argparse
import sys
import json

def restore_from_epi(video_path, output_folder):
    meta_path = video_path + ".json"
    if not os.path.exists(meta_path):
        print("ERRORE: Metadati .json mancanti.")
        sys.exit(1)
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    grid_u = meta['grid_u'] # Questa è l'altezza REALE utile
    grid_v = meta['grid_v']
    H = meta['orig_h']
    W = meta['orig_w']
    filenames = meta['filenames']
    
    # padding_h potrebbe non esistere nei vecchi file json, usiamo .get
    pad_h = meta.get('padding_h', 0)
    
    print(f"Metadati: Griglia {grid_u}x{grid_v}. Padding rilevato: {pad_h} pixels.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore video.")
        sys.exit(1)
        
    total_frames_epi = grid_v * H
    
    # Calcoliamo l'altezza che ci aspettiamo dal video (U + Padding)
    video_h = grid_u + pad_h
    
    print(f"Lettura video EPI...")
    
    # Buffer temporaneo con l'altezza "padded"
    epi_buffer = np.zeros((total_frames_epi, video_h, W, 3), dtype=np.uint8)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Se il frame letto è più grande/piccolo del previsto, avvisiamo
        if frame.shape != (video_h, W, 3):
            # Caso particolare: a volte FFmpeg aggiunge ulteriore padding automatico
            # Prendiamo solo la parte che ci serve
            if frame.shape[0] >= video_h and frame.shape[1] >= W:
                frame = frame[:video_h, :W, :]
            else:
                print(f"Errore critico dimensioni frame: {frame.shape}")
                break

        epi_buffer[frame_idx] = frame
        frame_idx += 1
        if frame_idx % 500 == 0: print(f"Letti {frame_idx}...", end='\r')

    cap.release()
    print(f"\nVideo caricato.")

    # --- RIMOZIONE PADDING ---
    if pad_h > 0:
        print(f"Rimozione di {pad_h} righe di padding...")
        # Tagliamo via le righe in eccesso in fondo
        epi_buffer = epi_buffer[:, :grid_u, :, :]
    
    # Ora epi_buffer ha altezza esattamente 'grid_u'
    
    # --- RICOSTRUZIONE ---
    print("Inversione trasformazione...")
    try:
        temp_5d = epi_buffer.reshape(grid_v, H, grid_u, W, 3)
    except ValueError:
        print("Errore reshape. Controlla che il numero di frame video corrisponda a V*H.")
        sys.exit(1)

    # Transpose inverso: da (V,H,U,W,C) a (U,V,H,W,C)
    lf_reconstructed = temp_5d.transpose(2, 0, 1, 3, 4)
    
    print(f"Salvataggio immagini in {output_folder}...")
    if not os.path.exists(output_folder): os.makedirs(output_folder)
        
    count = 0
    for v in range(grid_v):
        for u in range(grid_u):
            idx = v * grid_u + u
            if idx < len(filenames):
                name = filenames[idx]
                cv2.imwrite(os.path.join(output_folder, name), lf_reconstructed[u, v])
                count += 1
                
    print(f"Finito. {count} immagini ripristinate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("output_folder")
    args = parser.parse_args()
    
    restore_from_epi(args.input_video, args.output_folder)
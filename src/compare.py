import numpy as np
import cv2
import os
import glob
import argparse

def comparison(original_folder, decompressed_folder):

    def load_frames(folder):
        paths = sorted(glob.glob(os.path.join(folder, "*.png")))
        frames = [cv2.imread(p) for p in paths if cv2.imread(p) is not None]
        print(f"Caricati {len(frames)} frame da {folder}")
        return frames
    
    print("Confronto...")
    orig_frames = load_frames(original_folder)
    decomp_frames = load_frames(decompressed_folder)
    
    num_frames = min(len(orig_frames), len(decomp_frames))
    print(f"Frame da confrontare: {num_frames}")
    
    if num_frames == 0:
        print("Nessun frame da confrontare!")
        return
    
    total_diff = 0
    max_diff = 0
    
    for i in range(num_frames):
        # Differenza assoluta tra frame
        diff = cv2.absdiff(orig_frames[i], decomp_frames[i])
        mean_diff = np.mean(diff)
        total_diff += mean_diff
        
        if mean_diff > max_diff:
            max_diff = mean_diff
        
        if i < 3:  # Stampa primi 3
            print(f"Frame {i}: differenza media = {mean_diff:.2f}")
    
    avg_diff = total_diff / num_frames
    
    print(f"\nDifferenza media: {avg_diff:.2f}")
    print(f"Differenza massima: {max_diff:.2f}")
    
    if avg_diff < 1:
        print("PERFETTO - Nessuna differenza visibile")
    elif avg_diff < 5:
        print("OTTIMO - Differenze minime")
    elif avg_diff < 15:
        print("BUONO - Leggere differenze")
    else:
        print("DIFFERENZE SIGNIFICATIVE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Confronto ultra-semplice tra frame')
    parser.add_argument('original', help='Cartella con i frame originali')
    parser.add_argument('decompressed', help='Cartella con i frame decompressi')
    
    args = parser.parse_args()
    
    comparison(args.original, args.decompressed)
import os
import glob
import cv2
import numpy as np
from pathlib import Path

def analyze_lightfield_grid(folder_path):
    """
    Analizza una cartella di light field e suggerisce la griglia UxV ottimale
    """
    print(f"🔍 Analizzando cartella: {folder_path}")
    
    # Cerca tutti i file immagine
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
    file_paths = []
    for ext in extensions:
        file_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    file_paths.sort()
    total_files = len(file_paths)
    
    print(f"📁 File trovati: {total_files}")
    
    if total_files == 0:
        print("❌ Nessun file immagine trovato!")
        return None
    
    # Mostra primi file per capire il pattern
    print("\n📝 Primi 5 file:")
    for i, path in enumerate(file_paths[:5]):
        print(f"   {i+1}. {os.path.basename(path)}")
    
    # Trova tutte le possibili griglie fattorizzando il numero totale
    possible_grids = []
    for u in range(1, int(np.sqrt(total_files)) + 1):
        if total_files % u == 0:
            v = total_files // u
            possible_grids.append((u, v))
            # Aggiungi anche il simmetrico
            if u != v:
                possible_grids.append((v, u))
    
    # Ordina per griglie più "quadrate"
    possible_grids.sort(key=lambda x: abs(x[0] - x[1]))
    
    print(f"\n🔢 Possibili configurazioni griglia per {total_files} views:")
    for u, v in possible_grids:
        print(f"   {u}×{v} = {u*v} views")
    
    # Analisi dei nomi file per suggerimenti
    print(f"\n🎯 SUGGERIMENTI BASATI SUI NOMI FILE:")
    
    # Cerca pattern comuni nei nomi
    basenames = [os.path.basename(p) for p in file_paths]
    
    # Pattern tipici
    patterns_found = []
    
    # Pattern 1: view_u_v
    if any('_' in name and name.count('_') >= 2 for name in basenames):
        print("   📋 Pattern rilevato: view_u_v (es: view_01_02.png)")
        # Prova ad estrarre coordinate
        try:
            sample_name = basenames[0]
            parts = Path(sample_name).stem.split('_')
            if len(parts) >= 3:
                print(f"   Esempio: {sample_name} → u={parts[-2]}, v={parts[-1]}")
        except:
            pass
    
    # Pattern 2: numerazione sequenziale
    if all(any(c.isdigit() for c in name) for name in basenames):
        print("   📋 Pattern rilevato: Numerazione sequenziale")
    
    # Analisi dimensioni immagine per capire se è plenoptic
    try:
        first_img = cv2.imread(file_paths[0])
        h, w = first_img.shape[:2]
        print(f"\n📐 Dimensioni immagine: {w}×{h} px")
        
        # Plenoptic cameras tipicamente hanno dimensioni quadrate o particolari
        if w == h:
            print("   ℹ️  Immagine quadrata → possibile camera plenoptic")
        elif w > h * 1.5:
            print("   ℹ️  Formato panoramico → possibile camera array")
    except:
        pass
    
    # Griglie preferite per numeri comuni
    common_grids = {
        25: [(5, 5)],
        81: [(9, 9)],
        100: [(10, 10)],
        144: [(12, 12)],
        169: [(13, 13)],
        196: [(14, 14)],
        225: [(15, 15)],
        256: [(16, 16)],
        289: [(17, 17)],
        324: [(18, 18)],
        361: [(19, 19)],
        400: [(20, 20)]
    }
    
    print(f"\n🎲 GRIGLIA CONSIGLIATA:")
    if total_files in common_grids:
        recommended = common_grids[total_files][0]
        print(f"   ✅ **{recommended[0]}×{recommended[1]}** (configurazione standard)")
    else:
        # Scegli la griglia più "quadrata"
        recommended = possible_grids[0]
        print(f"   ✅ **{recommended[0]}×{recommended[1]}** (griglia più bilanciata)")
    
    # Mostra comando da usare
    print(f"\n💻 COMANDO DA USARE:")
    print(f"   python epi_compression.py {folder_path} --u {recommended[0]} --v {recommended[1]} -o output.mkv")
    
    return recommended

def verify_grid_with_thumbnails(folder_path, grid_u, grid_v):
    """
    Crea una thumbnail della griglia per verifica visiva
    """
    file_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    total_expected = grid_u * grid_v
    file_paths = file_paths[:total_expected]
    
    if len(file_paths) < total_expected:
        print(f"⚠️  Attenzione: servono {total_expected} views, trovate {len(file_paths)}")
        return
    
    # Leggi una immagine campione per le dimensioni
    sample = cv2.imread(file_paths[0])
    h, w = sample.shape[:2]
    
    # Crea thumbnail (ridimensiona per visualizzazione)
    thumb_scale = 0.1
    thumb_w, thumb_h = int(w * thumb_scale), int(h * thumb_scale)
    
    # Crea canvas per la griglia
    grid_img = np.zeros((thumb_h * grid_v, thumb_w * grid_u, 3), dtype=np.uint8)
    
    print(f"\n🖼️  Creazione thumbnail di verifica {grid_u}×{grid_v}...")
    
    for idx, path in enumerate(file_paths):
        if idx >= total_expected:
            break
            
        u = idx % grid_u
        v = idx // grid_u
        
        img = cv2.imread(path)
        if img is not None:
            thumb = cv2.resize(img, (thumb_w, thumb_h))
            grid_img[v*thumb_h:(v+1)*thumb_h, u*thumb_w:(u+1)*thumb_w] = thumb
    
    # Salva thumbnail
    output_path = "grid_verification.jpg"
    cv2.imwrite(output_path, grid_img)
    print(f"   📸 Thumbnail salvata: {output_path}")
    print("   🔍 Controlla che le immagini formino una griglia regolare!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizza light field e suggerisce griglia UxV')
    parser.add_argument('folder', help='Cartella contenente i frame della light field')
    parser.add_argument('--verify', action='store_true', help='Crea thumbnail di verifica')
    
    args = parser.parse_args()
    
    # Analisi principale
    grid = analyze_lightfield_grid(args.folder)
    
    # Verifica visiva opzionale
    if args.verify and grid:
        verify_grid_with_thumbnails(args.folder, grid[0], grid[1])
    
    print(f"\n🎉 Analisi completata! Usa --u {grid[0]} --v {grid[1]} nel tuo script EPI")
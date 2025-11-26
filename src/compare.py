import numpy as np
import cv2
import os
import glob
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def comparison(original_folder, decompressed_folder):
    
    def load_frames(folder):
        paths = sorted(glob.glob(os.path.join(folder, "*.png")))
        frames = [cv2.imread(p) for p in paths if cv2.imread(p) is not None]
        print(f"Caricati {len(frames)} frame da {folder}")
        return frames, [os.path.basename(p) for p in paths[:len(frames)]]
    
    print("CONFRONTO QUALITÀ - PSNR & SSIM")
    orig_frames, orig_names = load_frames(original_folder)
    decomp_frames, decomp_names = load_frames(decompressed_folder)
    
    num_frames = min(len(orig_frames), len(decomp_frames))
    print(f"Frame da confrontare: {num_frames}")
    
    if num_frames == 0:
        print("Nessun frame da confrontare!")
        return
    
    # Metriche
    psnr_values = []
    ssim_values = []
    mse_values = []
    
    print("\nANALISI FRAME-BY-FRAME:")
    print("-" * 60)
    
    for i in range(num_frames):
        orig = orig_frames[i].astype(np.float32)
        decomp = decomp_frames[i].astype(np.float32)
        
        # Calcola PSNR
        current_psnr = psnr(orig, decomp, data_range=255)
        psnr_values.append(current_psnr)
        
        # Calcola SSIM (media sui canali)
        ssim_channels = []
        for channel in range(3):  # BGR
            ssim_channel = ssim(orig[:,:,channel], decomp[:,:,channel], 
                               data_range=255)
            ssim_channels.append(ssim_channel)
        current_ssim = np.mean(ssim_channels)
        ssim_values.append(current_ssim)
        
        # Calcola MSE
        current_mse = np.mean((orig - decomp) ** 2)
        mse_values.append(current_mse)
        
        # Stampa primi 5 frame dettagliati
        if i < 5:
            frame_name = orig_names[i] if i < len(orig_names) else f"frame_{i}"
            print(f"  {frame_name}:")
            print(f"    PSNR: {current_psnr:7.2f} dB")
            print(f"    SSIM: {current_ssim:7.4f}")
            print(f"    MSE:  {current_mse:7.2f}")
    
    # Calcola statistiche
    psnr_mean = np.mean(psnr_values)
    psnr_std = np.std(psnr_values)
    psnr_min = np.min(psnr_values)
    psnr_max = np.max(psnr_values)
    
    ssim_mean = np.mean(ssim_values)
    ssim_std = np.std(ssim_values)
    ssim_min = np.min(ssim_values)
    ssim_max = np.max(ssim_values)
    
    mse_mean = np.mean(mse_values)
    
    print("-" * 60)
    print("\n RISULTATI FINALI:")
    print("=" * 40)
    
    # PSNR
    print(f" PSNR (Peak Signal-to-Noise Ratio):")
    print(f"   Media:    {psnr_mean:6.2f} dB")
    print(f"   Dev Std:  {psnr_std:6.2f} dB")
    print(f"   Min:      {psnr_min:6.2f} dB")
    print(f"   Max:      {psnr_max:6.2f} dB")
    
    # SSIM
    print(f"\n SSIM (Structural Similarity):")
    print(f"   Media:    {ssim_mean:8.6f}")
    print(f"   Dev Std:  {ssim_std:8.6f}")
    print(f"   Min:      {ssim_min:8.6f}")
    print(f"   Max:      {ssim_max:8.6f}")
    
    # MSE
    print(f"\nMSE (Mean Squared Error):")
    print(f"   Media:    {mse_mean:8.2f}")
    
    # Valutazione qualità
    print(f"\nVALUTAZIONE QUALITÀ:")
    print("=" * 40)
    
    # Valutazione PSNR
    if psnr_mean > 50:
        psnr_rating = "ECCELLENTE (quasi identico)"
    elif psnr_mean > 40:
        psnr_rating = "OTTIMO (perdita impercettibile)"
    elif psnr_mean > 30:
        psnr_rating = "BUONO (perdita appena percettibile)"
    elif psnr_mean > 20:
        psnr_rating = "ACCETTABILE (perdita visibile)"
    else:
        psnr_rating = "SCADENTE (forte degradazione)"
    
    # Valutazione SSIM
    if ssim_mean > 0.95:
        ssim_rating = "ECCELLENTE (strutture preservate)"
    elif ssim_mean > 0.90:
        ssim_rating = "OTTIMO (alta similarità strutturale)"
    elif ssim_mean > 0.80:
        ssim_rating = "BUONO (similarità accettabile)"
    elif ssim_mean > 0.70:
        ssim_rating = "DISCRETO (differenze strutturali)"
    else:
        ssim_rating = "SCADENTE (differenze marcate)"
    
    print(f"PSNR: {psnr_rating}")
    print(f"SSIM: {ssim_rating}")
    
    # Verifica lossless
    if psnr_mean > 60 and ssim_mean > 0.999:
        print("\n🎉 COMPRESSIONE LOSSLESS CONFERMATA!")
    elif mse_mean < 0.1:
        print("\n🎉 COMPRESSIONE QUASI-LOSSLESS!")
    
    return {
        'psnr_mean': psnr_mean,
        'psnr_std': psnr_std,
        'ssim_mean': ssim_mean, 
        'ssim_std': ssim_std,
        'mse_mean': mse_mean,
        'num_frames': num_frames
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Confronto qualità con metriche PSNR e SSIM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
ESEMPI:
  python compare_quality.py original/ decompressed/
  python compare_quality.py ../datasets/Fish/ decompressed_fish/
        
INTERPRETAZIONE:
  PSNR > 40 dB: Ottima qualità
  SSIM > 0.95:  Strutture ben preservate
  PSNR > 60 dB: Probabilmente lossless
        '''
    )
    parser.add_argument('original', help='Cartella con i frame originali')
    parser.add_argument('decompressed', help='Cartella con i frame decompressi')
    
    args = parser.parse_args()
    
    # Verifica esistenza cartelle
    if not os.path.exists(args.original):
        print(f"Cartella originale non trovata: {args.original}")
        sys.exit(1)
        
    if not os.path.exists(args.decompressed):
        print(f"Cartella decompressa non trovata: {args.decompressed}")
        sys.exit(1)
    
    results = comparison(args.original, args.decompressed)
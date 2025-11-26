#!/usr/bin/env python3
"""
Compressore Lossless per Light Field Images - Versione Corretta
Utilizzo: python fast_lightfield_compressor.py <comando> [parametri]
"""

import numpy as np
import zlib
import pickle
import os
import sys
import argparse
from PIL import Image
import glob
from typing import Tuple, List
from numba import jit, prange
import time

# Funzioni di supporto per Numba (devono essere definite a livello di modulo)
@jit(nopython=True, fastmath=True)
def clip_value(value: int, min_val: int, max_val: int) -> int:
    """Funzione clip ottimizzata per Numba"""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value

@jit(nopython=True, parallel=True, fastmath=True)
def compute_residuals_grayscale(light_field: np.ndarray) -> np.ndarray:
    """
    Calcolo ottimizzato dei residui per immagini in scala di grigi.
    """
    y_size, x_size, v_size, u_size = light_field.shape
    residuals = np.zeros((y_size, x_size, v_size, u_size), dtype=np.int16)
    lf_int32 = light_field.astype(np.int32)
    
    for u in prange(u_size):
        for v in range(v_size):
            for y in range(y_size):
                for x in range(x_size):
                    actual = lf_int32[y, x, v, u]
                    pred = 0
                    count = 0
                    
                    # Predizione spaziale
                    if x > 0:
                        pred += lf_int32[y, x-1, v, u]
                        count += 1
                    if y > 0:
                        pred += lf_int32[y-1, x, v, u]
                        count += 1
                    if x > 0 and y > 0:
                        pred += lf_int32[y-1, x-1, v, u]
                        count += 1
                    
                    # Predizione angolare
                    if u > 0:
                        pred += lf_int32[y, x, v, u-1]
                        count += 1
                    if v > 0:
                        pred += lf_int32[y, x, v-1, u]
                        count += 1
                    if u > 0 and v > 0:
                        pred += lf_int32[y, x, v-1, u-1]
                        count += 1
                    
                    if count > 0:
                        pred = pred // count
                    else:
                        pred = 0  # Primo pixel
                    
                    residuals[y, x, v, u] = actual - pred
    
    return residuals

@jit(nopython=True, parallel=True, fastmath=True)
def compute_residuals_rgb(light_field: np.ndarray) -> np.ndarray:
    """
    Calcolo ottimizzato dei residui per immagini RGB.
    """
    y_size, x_size, channels, v_size, u_size = light_field.shape
    residuals = np.zeros((y_size, x_size, channels, v_size, u_size), dtype=np.int16)
    lf_int32 = light_field.astype(np.int32)
    
    for u in prange(u_size):
        for v in range(v_size):
            for c in range(channels):
                for y in range(y_size):
                    for x in range(x_size):
                        actual = lf_int32[y, x, c, v, u]
                        pred = 0
                        count = 0
                        
                        # Predizione spaziale
                        if x > 0:
                            pred += lf_int32[y, x-1, c, v, u]
                            count += 1
                        if y > 0:
                            pred += lf_int32[y-1, x, c, v, u]
                            count += 1
                        if x > 0 and y > 0:
                            pred += lf_int32[y-1, x-1, c, v, u]
                            count += 1
                        
                        # Predizione angolare
                        if u > 0:
                            pred += lf_int32[y, x, c, v, u-1]
                            count += 1
                        if v > 0:
                            pred += lf_int32[y, x, c, v-1, u]
                            count += 1
                        if u > 0 and v > 0:
                            pred += lf_int32[y, x, c, v-1, u-1]
                            count += 1
                        
                        if count > 0:
                            pred = pred // count
                        else:
                            pred = 0  # Primo pixel
                        
                        residuals[y, x, c, v, u] = actual - pred
    
    return residuals

@jit(nopython=True, fastmath=True)
def reconstruct_grayscale(residuals: np.ndarray, first_pixel: int) -> np.ndarray:
    """
    Ricostruzione ottimizzata per immagini in scala di grigi.
    """
    y_size, x_size, v_size, u_size = residuals.shape
    reconstructed = np.zeros((y_size, x_size, v_size, u_size), dtype=np.uint8)
    
    # Inizializza il primo pixel
    reconstructed[0, 0, 0, 0] = first_pixel
    
    # Ricostruisci tutto il light field
    for u in range(u_size):
        for v in range(v_size):
            for y in range(y_size):
                for x in range(x_size):
                    # Salta il primo pixel che è già impostato
                    if u == 0 and v == 0 and y == 0 and x == 0:
                        continue
                    
                    pred = 0
                    count = 0
                    
                    # Predizione spaziale (usa solo pixel già ricostruiti)
                    if x > 0:
                        pred += reconstructed[y, x-1, v, u]
                        count += 1
                    if y > 0:
                        pred += reconstructed[y-1, x, v, u]
                        count += 1
                    if x > 0 and y > 0:
                        pred += reconstructed[y-1, x-1, v, u]
                        count += 1
                    
                    # Predizione angolare (usa solo viste già ricostruite)
                    if u > 0:
                        pred += reconstructed[y, x, v, u-1]
                        count += 1
                    if v > 0:
                        pred += reconstructed[y, x, v-1, u]
                        count += 1
                    if u > 0 and v > 0:
                        pred += reconstructed[y, x, v-1, u-1]
                        count += 1
                    
                    if count > 0:
                        pred = pred // count
                    else:
                        pred = 0
                    
                    value = pred + residuals[y, x, v, u]
                    reconstructed[y, x, v, u] = clip_value(value, 0, 255)
    
    return reconstructed

@jit(nopython=True, fastmath=True)
def reconstruct_rgb(residuals: np.ndarray, first_pixels: np.ndarray) -> np.ndarray:
    """
    Ricostruzione ottimizzata per immagini RGB.
    """
    y_size, x_size, channels, v_size, u_size = residuals.shape
    reconstructed = np.zeros((y_size, x_size, channels, v_size, u_size), dtype=np.uint8)
    
    # Inizializza il primo pixel per ogni canale
    for c in range(channels):
        reconstructed[0, 0, c, 0, 0] = first_pixels[c]
    
    # Ricostruisci tutto il light field
    for u in range(u_size):
        for v in range(v_size):
            for c in range(channels):
                for y in range(y_size):
                    for x in range(x_size):
                        # Salta il primo pixel che è già impostato
                        if u == 0 and v == 0 and y == 0 and x == 0:
                            continue
                        
                        pred = 0
                        count = 0
                        
                        # Predizione spaziale (usa solo pixel già ricostruiti)
                        if x > 0:
                            pred += reconstructed[y, x-1, c, v, u]
                            count += 1
                        if y > 0:
                            pred += reconstructed[y-1, x, c, v, u]
                            count += 1
                        if x > 0 and y > 0:
                            pred += reconstructed[y-1, x-1, c, v, u]
                            count += 1
                        
                        # Predizione angolare (usa solo viste già ricostruite)
                        if u > 0:
                            pred += reconstructed[y, x, c, v, u-1]
                            count += 1
                        if v > 0:
                            pred += reconstructed[y, x, c, v-1, u]
                            count += 1
                        if u > 0 and v > 0:
                            pred += reconstructed[y, x, c, v-1, u-1]
                            count += 1
                        
                        if count > 0:
                            pred = pred // count
                        else:
                            pred = 0
                        
                        value = pred + residuals[y, x, c, v, u]
                        reconstructed[y, x, c, v, u] = clip_value(value, 0, 255)
    
    return reconstructed

class FastLightFieldCompressor:
    def __init__(self):
        self.encoded_data = None
        
    def load_light_field_from_folder(self, folder_path: str, 
                                   spatial_shape: Tuple[int, int] = None,
                                   angular_grid: Tuple[int, int] = None,
                                   file_pattern: str = "Frame_%3d.png") -> np.ndarray:
        """
        Carica un light field da una cartella di immagini.
        """
        print(f"Caricamento light field da: {folder_path}")
        
        # Estrai il pattern per glob
        pattern = file_pattern.replace("%3d", "*")
        frame_files = sorted(glob.glob(os.path.join(folder_path, pattern)))
        
        if not frame_files:
            raise ValueError(f"Nessun file {pattern} trovato in {folder_path}")
        
        print(f"Trovati {len(frame_files)} frame")
        
        # Determina la griglia angolare
        if angular_grid is None:
            grid_size = int(np.sqrt(len(frame_files)))
            if grid_size * grid_size == len(frame_files):
                angular_grid = (grid_size, grid_size)
                print(f"Griglia angolare dedotta: {angular_grid}")
            else:
                found = False
                for i in range(2, min(20, len(frame_files))):
                    for j in range(2, min(20, len(frame_files))):
                        if i * j == len(frame_files):
                            angular_grid = (i, j)
                            print(f"Griglia angolare dedotta: {angular_grid}")
                            found = True
                            break
                    if found:
                        break
                if not found:
                    raise ValueError(f"Impossibile dedurre griglia angolare per {len(frame_files)} frame. Specifica --angular-grid manualmente.")
        
        v_views, u_views = angular_grid
        total_expected_frames = v_views * u_views
        
        if len(frame_files) != total_expected_frames:
            print(f"Avviso: Trovati {len(frame_files)} frame, ma la griglia {angular_grid} ne richiede {total_expected_frames}")
            if len(frame_files) < total_expected_frames:
                raise ValueError(f"Frame insufficienti: trovati {len(frame_files)}, necessari {total_expected_frames}")
            frame_files = frame_files[:total_expected_frames]
            print(f"Usati i primi {total_expected_frames} frame")
        
        # Carica la prima immagine per determinare le dimensioni
        first_image = Image.open(frame_files[0])
        if spatial_shape is None:
            spatial_shape = first_image.size[::-1]
        
        height, width = spatial_shape
        
        # Determina il tipo di dati
        if first_image.mode == 'L':
            dtype = np.uint8
        elif first_image.mode == 'RGB':
            dtype = np.uint8
        elif first_image.mode == 'I;16':
            dtype = np.uint16
        else:
            print(f"Avviso: Modalità immagine {first_image.mode} non supportata, converto in scala di grigi")
            dtype = np.uint8
        
        print(f"Dimensioni spaziali: {height}x{width}")
        print(f"Tipo dati: {dtype}")
        print(f"Griglia angolare: {v_views}x{u_views}")
        
        # Crea l'array 4D/5D per il light field
        if first_image.mode == 'RGB':
            light_field = np.zeros((height, width, 3, v_views, u_views), dtype=dtype)
        else:
            light_field = np.zeros((height, width, v_views, u_views), dtype=dtype)
        
        # Popola il light field
        frame_idx = 0
        for v in range(v_views):
            for u in range(u_views):
                if frame_idx < len(frame_files):
                    img = Image.open(frame_files[frame_idx])
                    
                    if img.mode != first_image.mode:
                        if first_image.mode == 'L':
                            img = img.convert('L')
                        elif first_image.mode == 'RGB':
                            img = img.convert('RGB')
                    
                    if img.size != (width, height):
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                    
                    img_array = np.array(img)
                    
                    if first_image.mode == 'RGB':
                        light_field[:, :, :, v, u] = img_array
                    else:
                        light_field[:, :, v, u] = img_array
                    
                    frame_idx += 1
        
        print(f"Light field caricato: {light_field.shape}")
        return light_field

    def compress(self, light_field: np.ndarray) -> bytes:
        """
        Compressione ottimizzata con Numba.
        """
        print("Avvio compressione ottimizzata...")
        start_time = time.time()
        
        is_rgb = len(light_field.shape) == 5
        
        if is_rgb:
            y_size, x_size, channels, v_size, u_size = light_field.shape
            print(f"Compressione immagine RGB: {y_size}x{x_size}x{channels}")
            
            # Calcola residui
            residuals = compute_residuals_rgb(light_field)
            
            # Prepara dati per la compressione
            compression_data = {
                'residuals': residuals,
                'original_shape': light_field.shape,
                'dtype': str(light_field.dtype),
                'is_rgb': True,
                'first_pixels': [light_field[0, 0, c, 0, 0] for c in range(3)]
            }
        else:
            y_size, x_size, v_size, u_size = light_field.shape
            print(f"Compressione immagine in scala di grigi: {y_size}x{x_size}")
            
            # Calcola residui
            residuals = compute_residuals_grayscale(light_field)
            
            # Prepara dati per la compressione
            compression_data = {
                'residuals': residuals,
                'original_shape': light_field.shape,
                'dtype': str(light_field.dtype),
                'is_rgb': False,
                'first_pixel': light_field[0, 0, 0, 0]
            }
        
        compression_time = time.time() - start_time
        print(f"Calcolo residui completato in {compression_time:.2f} secondi")
        
        # Compressione dei residui
        print("Compressione dati...")
        serialized_data = pickle.dumps(compression_data)
        compressed_data = zlib.compress(serialized_data, level=9)
        
        original_size = light_field.nbytes
        compressed_size = len(compressed_data)
        ratio = original_size / compressed_size
        
        total_time = time.time() - start_time
        
        print(f"Compressione completata in {total_time:.2f} secondi")
        print(f"Dimensione originale: {original_size:,} bytes")
        print(f"Dimensione compressa: {compressed_size:,} bytes")
        print(f"Rapporto di compressione: {ratio:.2f}x")
        print(f"Velocità: {original_size / total_time / 1e6:.2f} MB/s")
        
        return compressed_data

    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """
        Decompressione ottimizzata con Numba.
        """
        print("Avvio decompressione ottimizzata...")
        start_time = time.time()
        
        # Decomprimi e deserializza
        serialized_data = zlib.decompress(compressed_data)
        data = pickle.loads(serialized_data)
        
        residuals = data['residuals']
        original_shape = data['original_shape']
        is_rgb = data['is_rgb']
        
        # Ricostruzione
        if is_rgb:
            first_pixels = np.array(data['first_pixels'], dtype=np.uint8)
            reconstructed = reconstruct_rgb(residuals, first_pixels)
        else:
            first_pixel = data['first_pixel']
            reconstructed = reconstruct_grayscale(residuals, first_pixel)
        
        decompression_time = time.time() - start_time
        print(f"Decompressione completata in {decompression_time:.2f} secondi")
        
        return reconstructed

    def save_compressed(self, compressed_data: bytes, output_path: str):
        """Salva i dati compressi in un file."""
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
        print(f"Dati compressi salvati in: {output_path}")

    def load_compressed(self, input_path: str) -> bytes:
        """Carica dati compressi da un file."""
        with open(input_path, 'rb') as f:
            compressed_data = f.read()
        print(f"Dati compressi caricati da: {input_path}")
        return compressed_data

def main():
    parser = argparse.ArgumentParser(description='Compressore Lossless per Light Field Images - Versione Veloce')
    subparsers = parser.add_subparsers(dest='command', help='Comando da eseguire')
    
    # Parser per il comando compress
    compress_parser = subparsers.add_parser('compress', help='Comprimi un dataset di light field')
    compress_parser.add_argument('--input', '-i', required=True, help='Cartella di input con i frame')
    compress_parser.add_argument('--output', '-o', required=True, help='File di output compresso')
    compress_parser.add_argument('--angular-grid', '-g', type=int, nargs=2, 
                               help='Griglia angolare (es: 9 9 per 9x9 viste)')
    compress_parser.add_argument('--spatial-shape', '-s', type=int, nargs=2,
                               help='Dimensioni spaziali (es: 512 512 per 512x512 pixel)')
    compress_parser.add_argument('--file-pattern', '-p', default="Frame_%3d.png",
                               help='Pattern dei file (default: Frame_%3d.png)')
    
    # Parser per il comando decompress
    decompress_parser = subparsers.add_parser('decompress', help='Decomprimi un file light field')
    decompress_parser.add_argument('--input', '-i', required=True, help='File di input compresso')
    decompress_parser.add_argument('--output', '-o', required=True, help='Cartella di output per i frame decompressi')
    
    # Parser per il comando info
    info_parser = subparsers.add_parser('info', help='Mostra informazioni su un dataset')
    info_parser.add_argument('--input', '-i', required=True, help='Cartella con i frame o file compresso')
    
    # Parser per il comando benchmark
    benchmark_parser = subparsers.add_parser('benchmark', help='Test di velocità')
    benchmark_parser.add_argument('--input', '-i', required=True, help='Cartella di input con i frame')
    benchmark_parser.add_argument('--angular-grid', '-g', type=int, nargs=2, 
                                help='Griglia angolare (es: 9 9 per 9x9 viste)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    compressor = FastLightFieldCompressor()
    
    try:
        if args.command == 'compress':
            print("=== COMPRESSIONE LIGHT FIELD (VERSIONE VELOCE) ===")
            lf = compressor.load_light_field_from_folder(
                args.input,
                spatial_shape=args.spatial_shape,
                angular_grid=args.angular_grid,
                file_pattern=args.file_pattern
            )
            compressed_data = compressor.compress(lf)
            compressor.save_compressed(compressed_data, args.output)
            
        elif args.command == 'decompress':
            print("=== DECOMPRESSIONE LIGHT FIELD (VERSIONE VELOCE) ===")
            compressed_data = compressor.load_compressed(args.input)
            decompressed_lf = compressor.decompress(compressed_data)
            
            # Salva i frame decompressi
            os.makedirs(args.output, exist_ok=True)
            
            if len(decompressed_lf.shape) == 5:  # RGB
                y_size, x_size, channels, v_size, u_size = decompressed_lf.shape
                for v in range(v_size):
                    for u in range(u_size):
                        frame_idx = v * u_size + u
                        img_array = decompressed_lf[:, :, :, v, u]
                        img = Image.fromarray(img_array, 'RGB')
                        img.save(os.path.join(args.output, f"decompressed_frame_{frame_idx:03d}.png"))
            else:  # Grayscale
                y_size, x_size, v_size, u_size = decompressed_lf.shape
                for v in range(v_size):
                    for u in range(u_size):
                        frame_idx = v * u_size + u
                        img_array = decompressed_lf[:, :, v, u]
                        img = Image.fromarray(img_array, 'L')
                        img.save(os.path.join(args.output, f"decompressed_frame_{frame_idx:03d}.png"))
            
            print(f"Frame decompressi salvati in: {args.output}")
            
        elif args.command == 'info':
            if os.path.isdir(args.input):
                pattern = "Frame_*.png"
                frame_files = sorted(glob.glob(os.path.join(args.input, pattern)))
                if not frame_files:
                    pattern = "*.png"
                    frame_files = sorted(glob.glob(os.path.join(args.input, pattern)))
                
                print(f"Cartella: {args.input}")
                print(f"Numero di frame trovati: {len(frame_files)}")
                if frame_files:
                    first_img = Image.open(frame_files[0])
                    print(f"Dimensione frame: {first_img.size}")
                    print(f"Modalità: {first_img.mode}")
                    
                    print("\nPossibili griglie angolari:")
                    for i in range(2, min(10, len(frame_files) + 1)):
                        for j in range(2, min(10, len(frame_files) + 1)):
                            if i * j == len(frame_files):
                                print(f"  {i}x{j}")
            else:
                compressed_data = compressor.load_compressed(args.input)
                serialized_data = zlib.decompress(compressed_data)
                data = pickle.loads(serialized_data)
                print(f"File compresso: {args.input}")
                print(f"Forma originale: {data['original_shape']}")
                print(f"Tipo dati: {data['dtype']}")
                print(f"RGB: {data['is_rgb']}")
                
        elif args.command == 'benchmark':
            print("=== BENCHMARK VELOCITÀ ===")
            lf = compressor.load_light_field_from_folder(
                args.input,
                angular_grid=args.angular_grid
            )
            
            # Test compressione
            start_time = time.time()
            compressed_data = compressor.compress(lf)
            compression_time = time.time() - start_time
            
            # Test decompressione
            start_time = time.time()
            decompressed_lf = compressor.decompress(compressed_data)
            decompression_time = time.time() - start_time
            
            # Verifica lossless
            is_lossless = np.array_equal(lf, decompressed_lf)
            
            print(f"\n=== RISULTATI BENCHMARK ===")
            print(f"Lossless: {is_lossless}")
            print(f"Tempo compressione: {compression_time:.2f}s")
            print(f"Tempo decompressione: {decompression_time:.2f}s")
            print(f"Velocità compressione: {lf.nbytes / compression_time / 1e6:.2f} MB/s")
            print(f"Velocità decompressione: {lf.nbytes / decompression_time / 1e6:.2f} MB/s")
            print(f"Rapporto compressione: {lf.nbytes / len(compressed_data):.2f}x")
                
    except Exception as e:
        print(f"ERRORE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
import subprocess, time, av
from supported_algos import Algorithm
import os

class Compressor:
    @staticmethod
    def compress(self,algorithm: Algorithm, dataset_path: str, output_path: str) -> None:

        start_time = time.time()
        
        cmd = [
            "ffmpeg",
            "-framerate", "120",
            "-i", os.path.join(dataset_path, "Frame_%03d.png"),
            "-c:v", algorithm.codec
        ]
 
        cmd += algorithm.extra_args
        output_file = f"{output_path}/{algorithm.name}.{algorithm.output_format}"
        cmd.append(output_file)

        subprocess.run(cmd, check=True)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Compression with {algorithm.name} completed in {elapsed_time:.2f}")



class Decompressor:
    @staticmethod
    def decompress(compressed_file_path: str, output_path: str, number_of_frames: int = -1) -> None:
        start_time = time.time()

        # crea la cartella di output se non esiste
        os.makedirs(output_path, exist_ok=True)

        container = av.open(compressed_file_path, mode="r")
        stream = container.streams.video[0]

        count = 0
        for frame in container.decode(stream):
            if number_of_frames != -1 and count >= number_of_frames:
                break

            filename = os.path.join(output_path, f"Frame_{count:03d}.png")
            frame.to_image().save(filename)
            count += 1

        end_time = time.time()
        print(f"Decompression completed in {end_time - start_time:.2f} seconds.")



def get_extra_args():
    return []
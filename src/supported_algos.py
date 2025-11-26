from enum import Enum

class Algorithm(Enum):
    HEVC = ("libx265", "mp4", [
        "-x265-params", "lossless=1",
        "-pix_fmt", "yuv444p"  
    ])
    AV1 = ("libaom-av1", "mkv", ["-aom-params", "lossless=1"])
    VP9 = ("libvpx-vp9", "webm", ["-lossless", "1"])
    HUFFYUV = ("huffyuv", "avi", [])
    UTVIDEO = ("utvideo", "avi", [])
    FFV1 = ("ffv1", "mkv", ["-level", "3", "-coder", "2"])
    HEVC_VS = ("libx265", "mp4", ["-crf", "3"])
    AV1_VS = ("libaom-av1", "mkv", ["-crf", "3"])
    VP9_VS = ("libvpx-vp9", "webm", ["-crf", "3"])
    DIRAC = ("dirac", "mkv", [])
    MS_RLE = ("msrle", "avi", [])

    def __init__(self, codec: str, output_format: str, extra_args: list[str]):
        self.codec = codec
        self.output_format = output_format
        self.extra_args = extra_args

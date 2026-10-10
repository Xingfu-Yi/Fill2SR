import PIL
from PIL import Image
import PIL.Image
import numpy as np
import torch
from typing import Union, List

ASPECT_RATIOS = [16 / 9, 4 / 3, 1 / 1, 3 / 4, 9 / 16]
SIZE_LIST = [
    1536,
    2048,
    2304,
    2560,
    2688,
    2880,
    3008,
    3072,
    3200,
    3328,
    3456,
    3584,
    3712,
    3840,
    4096,
    4352,
    4608,
    4864,
    5120,
    5376,
    5632,
    5888,
    6144,
    6400,
    6656,
    6912,
    7168,
    7424,
    7680,
    7936,
    8192,
]

PIL_INTERPOLATION = {
    "linear": PIL.Image.Resampling.BILINEAR,
    "bilinear": PIL.Image.Resampling.BILINEAR,
    "bicubic": PIL.Image.Resampling.BICUBIC,
    "lanczos": PIL.Image.Resampling.LANCZOS,
    "nearest": PIL.Image.Resampling.NEAREST,
}


def get_image_resolutions(width: int, height: int):
    def closest_aspect_ratio(width: int, height: int, aspect_ratios: List) -> float:
        # Calculate actual aspect ratio of the image
        actual_ratio = width / height

        # Find the closest ratio from the provided list
        closest_ratio = min(aspect_ratios, key=lambda r: abs(r - actual_ratio))

        return closest_ratio

    def get_resolutions(aspect_ratio: list, size_list: list, max_width: int, max_height: int) -> list:
        """
        Generate a list of (width, height) resolution pairs based on aspect ratios and max resolution.

        Args:
            aspect_ratio (list): List of available width/height ratios (e.g., 16/9, 1.0, etc.).
            size_list (list): List of side resolutions (usually interpreted as width).

        Returns:
            list: A sorted list of (width, height) tuples where both dimensions are integers,
                and height does not exceed the max resolution.
        """
        resolutions = set()

        for res in size_list:
            for ratio in aspect_ratio:
                width = res
                height = width / ratio
                if height.is_integer() and height % 32 == 0 and width % 32 == 0:
                    # if width <= max_width and height <= max_height:
                    resolutions.add((int(width), int(height)))

                height = res
                width = height * ratio
                if width.is_integer() and height % 32 == 0 and width % 32 == 0:
                    # if width <= max_width and height <= max_height:
                    resolutions.add((int(width), int(height)))

        # Sort by width first, then height
        resolutions = sorted(resolutions, key=lambda x: x[0] * x[1], reverse=True)

        return resolutions

    closet_ratio = closest_aspect_ratio(width=width, height=height, aspect_ratios=ASPECT_RATIOS)
    resolutions = get_resolutions(
        aspect_ratio=[closet_ratio],
        size_list=SIZE_LIST,
        max_width=width,
        max_height=height,
    )
    return closet_ratio, resolutions


class ImageProcessor:
    def __init__(self):
        pass

    def _resize_and_fill(
        self,
        image: PIL.Image.Image,
        width: int,
        height: int,
    ) -> PIL.Image.Image:
        r"""
        Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center
        the image within the dimensions, filling empty with data from image.

        Args:
            image (`PIL.Image.Image`):
                The image to resize and fill.
            width (`int`):
                The width to resize the image to.
            height (`int`):
                The height to resize the image to.

        Returns:
            `PIL.Image.Image`:
                The resized and filled image.
        """

        ratio = width / height
        src_ratio = image.width / image.height

        src_w = width if ratio < src_ratio else image.width * height // image.height
        src_h = height if ratio >= src_ratio else image.height * width // image.width

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION["lanczos"])
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(
                    resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                    box=(0, fill_height + src_h),
                )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(
                    resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                    box=(fill_width + src_w, 0),
                )

        return res

    def _resize_and_crop(
        self,
        image: PIL.Image.Image,
        width: int,
        height: int,
    ) -> PIL.Image.Image:
        r"""
        Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center
        the image within the dimensions, cropping the excess.

        Args:
            image (`PIL.Image.Image`):
                The image to resize and crop.
            width (`int`):
                The width to resize the image to.
            height (`int`):
                The height to resize the image to.

        Returns:
            `PIL.Image.Image`:
                The resized and cropped image.
        """
        ratio = width / height
        src_ratio = image.width / image.height

        src_w = width if ratio > src_ratio else image.width * height // image.height
        src_h = height if ratio <= src_ratio else image.height * width // image.width

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION["lanczos"])
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        return res

    def resize(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        height: int,
        width: int,
        resize_mode: str = "default",  # "default", "fill", "crop"
    ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        """
        Resize image.

        Args:
            image (`PIL.Image.Image`, `np.ndarray` or `torch.Tensor`):
                The image input, can be a PIL image, numpy array or pytorch tensor.
            height (`int`):
                The height to resize to.
            width (`int`):
                The width to resize to.
            resize_mode (`str`, *optional*, defaults to `default`):
                The resize mode to use, can be one of `default` or `fill`. If `default`, will resize the image to fit
                within the specified width and height, and it may not maintaining the original aspect ratio. If `fill`,
                will resize the image to fit within the specified width and height, maintaining the aspect ratio, and
                then center the image within the dimensions, filling empty with data from image. If `crop`, will resize
                the image to fit within the specified width and height, maintaining the aspect ratio, and then center
                the image within the dimensions, cropping the excess. Note that resize_mode `fill` and `crop` are only
                supported for PIL image input.

        Returns:
            `PIL.Image.Image`, `np.ndarray` or `torch.Tensor`:
                The resized image.
        """
        if resize_mode != "default" and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"Only PIL image input is supported for resize_mode {resize_mode}")
        if isinstance(image, PIL.Image.Image):
            if resize_mode == "default":
                image = image.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
            elif resize_mode == "fill":
                image = self._resize_and_fill(image, width, height)
            elif resize_mode == "crop":
                image = self._resize_and_crop(image, width, height)
            else:
                raise ValueError(f"resize_mode {resize_mode} is not supported")

        elif isinstance(image, torch.Tensor):
            image = torch.nn.functional.interpolate(
                image,
                size=(height, width),
            )
        elif isinstance(image, np.ndarray):
            image = self.numpy_to_pt(image)
            image = torch.nn.functional.interpolate(
                image,
                size=(height, width),
            )
            image = self.pt_to_numpy(image)
        return image

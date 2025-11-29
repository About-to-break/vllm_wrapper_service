from io import BytesIO
from typing import Optional, List, Dict, Any
from PIL import Image, UnidentifiedImageError
import base64
from openai import OpenAI
import prompts
import logging

class OpenAiClient:
    def __init__(
        self,
        logger: logging.Logger,
        base_url: str,
        api_key: Optional[str] = None,
    ):
        self.logger = logger
        self.api_key = api_key or "EMPTY"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=60.0,
            max_retries=2,
        )

    @staticmethod
    def b64_convert_image(image_b: bytes, mime_type: str = "jpeg") -> str:
        """Convert image bytes to a base64 data URL."""
        encoded = base64.b64encode(image_b).decode("ascii")
        return f"data:image/{mime_type};base64,{encoded}"

    @staticmethod
    def _compress_image(
        image_b: bytes,
        compression_max_size: int = 1024,
        format: str = "JPEG",
        quality: int = 85,
    ) -> tuple[bytes, str]:
        """
        Compress an image to fit within the given maximum dimension.
        Returns the compressed image bytes and the corresponding MIME type.
        """
        try:
            img = Image.open(BytesIO(image_b))
        except UnidentifiedImageError as e:
            raise ValueError("Failed to identify image file: corrupted or unsupported format.") from e

        if img.mode in ("RGBA", "LA", "P") and format.upper() == "JPEG":
            # JPEG does not support transparency — flatten to RGB with white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode.endswith("A") else None)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        img.thumbnail((compression_max_size, compression_max_size), Image.LANCZOS)

        buffer = BytesIO()
        save_format = "JPEG"
        mime_type = "jpeg"

        if format.upper() == "PNG":
            save_format = "PNG"
            mime_type = "png"
            img.save(buffer, format=save_format, optimize=True)
        else:
            img.save(buffer, format=save_format, quality=quality, optimize=True)

        return buffer.getvalue(), mime_type

    def _generate_vl_message(
        self,
        image_b: bytes,
        compression_max_size: int = 1024,
        custom_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a multimodal message for a vision-language model.
        """
        try:
            compressed_bytes, mime_type = self._compress_image(
                image_b, compression_max_size=compression_max_size
            )
        except ValueError as e:
            self.logger.error(f"Image compression failed: {e}")
            raise

        b64_str = self.b64_convert_image(compressed_bytes, mime_type=mime_type)
        message = prompts.image_message(
            b64_img_str=b64_str,
            custom_prompt=custom_prompt,
        )
        return message

    def vl_request(
        self,
        image_b: bytes,
        model: str,
        compression_max_size: int = 1024,
        temperature: float = 0.1,
        custom_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """
        Send a vision-language request to the model (e.g., Qwen-VL).
        Returns the model's textual response or None on failure.
        """
        if not isinstance(image_b, bytes) or len(image_b) == 0:
            self.logger.error("Provided image bytes are empty or invalid.")
            return None

        try:
            messages = self._generate_vl_message(
                image_b=image_b,
                compression_max_size=compression_max_size,
                custom_prompt=custom_prompt,
            )

            request_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**request_kwargs)

            if not response.choices:
                self.logger.warning("Received empty response from model (no choices returned).")
                return None

            content = response.choices[0].message.content
            if content is None:
                self.logger.warning("Model returned a message with null content.")
                return None

            return content.strip()

        except Exception as e:
            self.logger.error(f"VL request failed: {e}", exc_info=True)
            return None
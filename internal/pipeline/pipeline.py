import logging
import ast

from minio_tools import MinioClient
from internal.openai_api.openai_client import OpenAiVlClient


class MainPipeline:
    def __init__(self,
                 logger: logging.Logger,
                 minio_client: MinioClient,
                 openai_url: str,
                 openai_api_key: str = None,
                 ):
        self.logger = logger
        self.logger.info("Initializing main service pipeline...")
        self.minio_client = minio_client
        self.openai_client = OpenAiVlClient(
            logger=self.logger,
            api_key=openai_api_key,
            base_url=openai_url
        )
        self.logger.info("Main service pipeline initialized successfully.")
    @staticmethod
    def _decode_body(body: bytes) -> dict:
        text = body.decode("utf-8")
        parsed_text = ast.literal_eval(text)
        return parsed_text

    def run(self, body: bytes, model: str):
        try:
            parsed_body = self._decode_body(body)
            file_path, uuid = parsed_body.values()

            self.logger.debug(f"Decoded body with uuid {uuid}")

            image = self.minio_client.download_file(file_path)
            if image is None:
                raise Exception("Failed to download image")

            response, compression_k = self.openai_client.img_request(
                image_b=image,
                model=model,
            )

            if response is None and compression_k is None:
                raise Exception("Failed to get LLM answer")

            self.logger.debug(f"Decoded body with response {response}")

        except Exception as e:
            self.logger.error(f"Pipeline exception: {e}")


    def test_run(self, body: bytes):
        self.logger.warning(f"Decoded body with body {body}")





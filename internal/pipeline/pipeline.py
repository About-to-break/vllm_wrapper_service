import logging
import ast

from minio_tools import MinioClient
from internal.roi_processor import roi


class MainPipeline:
    def __init__(self,
                 logger: logging.Logger,
                 minio_client: MinioClient,
                 roi_device: str,
                 model: str,
                 roi_target_class: str,
                 ):
        self.logger = logger
        self.logger.info("Initializing main service pipeline...")
        self.model = model
        self.minio_client = minio_client
        self.roi = roi.ROI(
            logger=self.logger,
            model=self.model,
            device=roi_device,
            target_class=roi_target_class,
        )
        self.logger.info("Main service pipeline initialized successfully.")

    @staticmethod
    def _decode_body(body: bytes) -> dict:
        text = body.decode("utf-8")
        parsed_text = ast.literal_eval(text)
        return parsed_text

    def run(self, body: bytes):
        try:
            parsed_body = self._decode_body(body)
            file_path, uuid = parsed_body.values()

            self.logger.debug(f"Decoded body with uuid {uuid}")

            image = self.minio_client.download_file(file_path)
            if image is None:
                raise Exception("Failed to download image")
            else:
                self.logger.debug(f"Image successfully downloaded from minio")

            response = self.roi.get_rois(image_b=image)

            if response is None:
                raise Exception("Failed to get ROI detected bboxes")

            self.logger.debug(f"Acquired clean ROI bboxes response: {response}")

            # TODO: перессылка в rabbit

        except Exception as e:
            self.logger.error(f"Pipeline exception: {e}")

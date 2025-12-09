import PIL
from ultralytics import YOLO
from PIL import Image
import io
import logging


class ROI:
    def __init__(self,
                 logger: logging.Logger,
                 model: str,
                 target_class: str,
                 device: str = None,
                 ):

        self.logger = logger

        try:
            self.model = YOLO(model)

            if device is None:
                self.device = 'cpu'
            else:
                self.device = device

                # In case if different model is used. Anyway it's used for aliases of 'frame' in mine
                self.target_index = None
                for idx, name in self.model.names.items():
                    if name == target_class:
                        self.target_index = idx
                        break

                if self.target_index is None:
                    raise Exception(f"No target class {target_class} found in model")

        except Exception as e:
            self.logger.error(f"Failed to initialize ROI: {e}")

    @staticmethod
    def _img_bytes_to_pil(image_b: bytes) -> PIL.Image.Image:
        return Image.open(io.BytesIO(image_b))

    def _filter_frames(self, results: list) -> list:
        frame_boxes = [box for box in results[0].boxes if int(box.cls[0]) == self.target_index]
        return frame_boxes

    def get_regions(self, image_b: bytes, **kwargs):
        default_params = {
            "half": True,
            "imgsz": 1024,
            "conf": 0.25,
            "save": False,
            "iou": 0.1,
        }

        params = {**default_params, **kwargs}

        self.logger.debug("Running frame recognition...")

        try:
            image = self._img_bytes_to_pil(image_b)
            results = self.model.predict(source=image,
                                         device=self.device,
                                         **params)
            return results
        except Exception as e:
            self.logger.error(f"Failed frame recognition: {e}")

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

        self.logger.info("Initializing ROI detector")

        try:
            self.model = YOLO(model)

            if device is None:
                self.device = 'cpu'
            else:
                self.device = device

                # In case if different model is used. Anyway it's meant for aliases of 'frame' class in mine one
                self.target_index = None
                for idx, name in self.model.names.items():
                    if name == target_class:
                        self.target_index = idx
                        break

                if self.target_index is None:
                    raise Exception(f"No target class {target_class} found in model")

                self.logger.info("Initialization ROI detector successfully done")
        except Exception as e:
            self.logger.error(f"Failed to initialize ROI: {e}")

    @staticmethod
    def _img_bytes_to_pil(image_b: bytes) -> PIL.Image.Image:
        return Image.open(io.BytesIO(image_b))

    def _filter_frames(self, results: list):
        frame_boxes = [box for box in results[0].boxes if int(box.cls[0]) == self.target_index]
        results[0].boxes = frame_boxes
        return results[0].boxes

    @staticmethod
    def _tidy_boxes(boxes):
        pretty_boxes = []
        for box in boxes:
            pretty_boxes.append([float(x) for x in box.xyxy[0]])

        return pretty_boxes

    # Not mainly for use in pipeline, better stick to self.get_rois()
    def get_raw_regions(self, image, **kwargs):
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
            results = self.model.predict(source=image,
                                         device=self.device,
                                         **params)
            return results
        except Exception as e:
            self.logger.error(f"Failed frame recognition: {e}")


    def get_rois(self, image_b: bytes, **kwargs):
        # Here we do all the stuff to return a bboxes markup with no empty space on img.
        # My model currently shows kind of good detection but looses far rims of frames.
        # So I came up that full markup likely will be profitable for later text detecting.

        self.logger.debug("Getting ROIs...")
        try:
            # First we get clean bboxes of detected frames

            image = self._img_bytes_to_pil(image_b)
            regions = self.get_raw_regions(image, **kwargs)
            boxes = self._filter_frames(regions)
            cleaned_boxes = self._tidy_boxes(boxes)
            # Now we will modify bboxes to fill all the gaps on img
            # In fact still testing if its crucial...
            # w, h = image.size

            return cleaned_boxes

        except Exception as e:
            self.logger.error(f"Failed to get ROIs: {e}")


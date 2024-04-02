
import json

import os, importlib

import numpy as np
import arcpy


def get_available_device(max_memory=0.8):
    """
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0 
    """
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available


features = {
    "displayFieldName": "",
    "fieldAliases": {"FID": "FID", "Class": "Class", "Confidence": "Confidence"},
    "geometryType": "esriGeometryPolygon",
    "fields": [
        {"name": "FID", "type": "esriFieldTypeOID", "alias": "FID"},
        {"name": "Class", "type": "esriFieldTypeString", "alias": "Class"},
        {"name": "Confidence", "type": "esriFieldTypeDouble", "alias": "Confidence"},
    ],
    "features": [],
}

fields = {
    "fields": [
        {"name": "OID", "type": "esriFieldTypeOID", "alias": "OID"},
        {"name": "Class", "type": "esriFieldTypeString", "alias": "Class"},
        {"name": "Confidence", "type": "esriFieldTypeDouble", "alias": "Confidence"},
        {"name": "Shape", "type": "esriFieldTypeGeometry", "alias": "Shape"},
    ]
}


class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4


class ArcGISObjectDetector:
    def __init__(self):
        self.name = "Object Detector"
        self.description = "This python raster function applies deep learning model to detect objects in imagery"

    def initialize(self, **kwargs):
        if "model" not in kwargs:
            return

        model = kwargs["model"]
        model_as_file = True
        try:
            with open(model, "r") as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        framework = self.json_info["Framework"]
        if "ModelConfiguration" in self.json_info:
            if isinstance(self.json_info["ModelConfiguration"], str):
                ChildModelDetector = getattr(
                    importlib.import_module(
                        "{}.{}".format(framework, self.json_info["ModelConfiguration"])
                    ),
                    "ChildObjectDetector",
                )
            else:
                ChildModelDetector = getattr(
                    importlib.import_module(
                        "{}.{}".format(
                            framework, self.json_info["ModelConfiguration"]["Name"]
                        )
                    ),
                    "ChildObjectDetector",
                )
        else:
            raise Exception("Invalid model configuration")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        if "device" in kwargs:
            device = kwargs["device"]
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception(
                        "PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials"
                    )
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"

        self.child_object_detector = ChildModelDetector()
        self.child_object_detector.initialize(model, model_as_file)

    def getParameterInfo(self):
        required_parameters = [
            {
                "name": "raster",
                "dataType": "raster",
                "required": True,
                "displayName": "Raster",
                "description": "Input Raster",
            },
            {
                "name": "model",
                "dataType": "string",
                "required": True,
                "displayName": "Input Model Definition (EMD) File",
                "description": "Input model definition (EMD) JSON file",
            },
            {
                "name": "device",
                "dataType": "numeric",
                "required": False,
                "displayName": "Device ID",
                "description": "Device ID",
            },
        ]
        parameter_info = self.child_object_detector.getParameterInfo(
            required_parameters
        )
        parameter_info.extend(
            [
                {
                    "name": "test_time_augmentation",
                    "dataType": "string",
                    "required": False,
                    "value": "False"
                    if "test_time_augmentation" not in self.json_info
                    else str(self.json_info["test_time_augmentation"]),
                    "displayName": "Perform test time augmentation while predicting",
                    "description": "If True, will merge predictions from flipped and rotated images.",
                }
            ]
        )
        return parameter_info

    def getConfiguration(self, **scalars):
        configuration = self.child_object_detector.getConfiguration(**scalars)
        if "DataRange" in self.json_info:
            configuration["dataRange"] = tuple(self.json_info["DataRange"])
        configuration["inheritProperties"] = 2 | 4 | 8
        configuration["inputMask"] = True
        self.use_tta = scalars.get("test_time_augmentation", "false").lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]
        self.nms_overlap = float(scalars.get("nms_overlap", 0.1))
        return configuration

    def getFields(self):
        return json.dumps(fields)

    def getGeometryType(self):
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):
        # set pixel values in invalid areas to 0
        raster_mask = pixelBlocks["raster_mask"]
        raster_pixels = pixelBlocks["raster_pixels"]
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks["raster_pixels"] = raster_pixels

        polygon_list, scores, classes = self.tta_detect_objects(**pixelBlocks)

        features["features"] = []
        for i in range(len(polygon_list)):
            rings = [[]]
            for j in range(polygon_list[i].shape[0]):
                rings[0].append([polygon_list[i][j][1], polygon_list[i][j][0]])

            features["features"].append(
                {
                    "attributes": {
                        "OID": i + 1,
                        "Class": self.json_info["Classes"][classes[i] - 1]["Name"],
                        "Confidence": scores[i],
                    },
                    "geometry": {"rings": rings},
                }
            )

        return {"output_vectors": json.dumps(features)}

    def tta_detect_objects(self, **pixelBlocks):
        import torch
        from fastai.vision.transform import dihedral_affine, rotate
        from fastai.vision import Image

        input_image = pixelBlocks["raster_pixels"].astype(np.float32)

        tile_size = input_image.shape[1]
        pad = self.child_object_detector.padding

        allboxes = torch.empty(0,4)
        allclasses = []
        allscores = torch.empty(0)

        boxes_list, scores_list, labels_list = [], [], []
        transforms = [0]

        if self.use_tta:
            if self.json_info["ImageSpaceUsed"] == "MAP_SPACE":
                transforms = list(range(8))
            else:
                transforms = [
                    0,
                    2,
                ]  # no vertical flips for pixel space (oriented imagery)

        for k in transforms:
            out = dihedral_affine(Image(torch.tensor(input_image.copy() / 256.0)), k)
            pixelBlocksCopy = pixelBlocks.copy()
            pixelBlocksCopy["raster_pixels"] = (out.data * 256).numpy()
            polygons, scores, classes = self.detect_objects(**pixelBlocksCopy)

            bboxes = self.get_img_bbox(tile_size, polygons, scores, classes)
            if bboxes is not None:
                fixed_img_bboxes = dihedral_affine(bboxes, k)
                if k == 5 or k == 6:
                    fixed_img_bboxes = rotate(fixed_img_bboxes, 180)

                allboxes = torch.cat([allboxes, (fixed_img_bboxes.data[0]+1) / 2.0])
                allclasses = allclasses + fixed_img_bboxes.data[1].tolist()
                allscores = np.concatenate([allscores, torch.tensor(scores) * 0.01])

                boxes_list.append((fixed_img_bboxes.data[0] + 1) / 2.0)
                scores_list.append(torch.tensor(scores) * 0.01)  # normalize to [0,1]
                labels_list.append(fixed_img_bboxes.data[1].tolist())

        try:
            from ensemble_boxes import weighted_boxes_fusion

            iou_thr = self.nms_overlap
            skip_box_thr = 0.0001

            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        except:
            import warnings

            warnings.warn("Unable to perform weighted boxes fusion... use NMS")
            boxes, scores, labels = np.array(allboxes), allscores, np.array(allclasses)

        bboxes = boxes * tile_size - pad
        polygons = self.convert_bounding_boxes_to_coord_list(bboxes)

        return polygons, np.array(scores * 100).astype(float), labels.astype(int)

    def get_img_bbox(self, tile_size, polygons, scores, classes):
        from fastai.vision import ImageBBox

        pad = self.child_object_detector.padding
        bboxes = []
        for i, polygon in enumerate(polygons):
            x1, y1 = np.around(polygon).astype(int)[0]
            x2, y2 = np.around(polygon).astype(int)[2]
            bboxes.append([x1 + pad, y1 + pad, x2 + pad, y2 + pad])
        n = len(bboxes)
        if n > 0:
            return ImageBBox.create(
                tile_size,
                tile_size,
                bboxes,
                labels=classes,
                classes=["Background"] + [x["Name"] for x in self.json_info["Classes"]],
            )
        else:
            return None

    def convert_bounding_boxes_to_coord_list(self, bounding_boxes):
        """
        convert bounding box numpy array to python list of point arrays
        :param bounding_boxes: numpy array of shape [n, 4]
        :return: python array of point numpy arrays, each point array is in shape [4,2]
        """
        num_bounding_boxes = bounding_boxes.shape[0]
        bounding_box_coord_list = []
        for i in range(num_bounding_boxes):
            coord_array = np.empty(shape=(4, 2), dtype=float)
            coord_array[0][0] = bounding_boxes[i][0]
            coord_array[0][1] = bounding_boxes[i][1]

            coord_array[1][0] = bounding_boxes[i][0]
            coord_array[1][1] = bounding_boxes[i][3]

            coord_array[2][0] = bounding_boxes[i][2]
            coord_array[2][1] = bounding_boxes[i][3]

            coord_array[3][0] = bounding_boxes[i][2]
            coord_array[3][1] = bounding_boxes[i][1]

            bounding_box_coord_list.append(coord_array)

        return bounding_box_coord_list

    def detect_objects(self, **pixelBlocks):
        polygon_list, scores, classes = self.child_object_detector.vectorize(
            **pixelBlocks
        )

        padding = self.child_object_detector.padding
        keep_polygon = []
        keep_scores = []
        keep_classes = []

        chip_sz = self.json_info["ImageHeight"]

        for idx, polygon in enumerate(polygon_list):
            centroid = polygon.mean(0)
            i, j = int(centroid[0]) // chip_sz, int(centroid[1]) // chip_sz
            x, y = int(centroid[0]) % chip_sz, int(centroid[1]) % chip_sz

            x1, y1 = polygon[0]
            x2, y2 = polygon[2]

            # fix polygon by removing padded regions
            polygon[:, 0] = polygon[:, 0] - (2 * i + 1) * padding
            polygon[:, 1] = polygon[:, 1] - (2 * j + 1) * padding

            X1, Y1, X2, Y2 = (
                i * chip_sz,
                j * chip_sz,
                (i + 1) * chip_sz,
                (j + 1) * chip_sz,
            )
            t = 2.0  # within 2 pixels of edge

            # if centroid not in center, reduce confidence
            # so box can be filtered out during NMS
            if (
                x < padding
                or x > chip_sz - padding
                or y < padding
                and y > chip_sz - padding
            ):

                scores[idx] = (self.child_object_detector.thres * 100) + scores[
                    idx
                ] * 0.01

            # if not excluded due to touching edge of tile
            if not (
                self.child_object_detector.filter_outer_padding_detections
                and any(
                    [
                        abs(X1 - x1) < t,
                        abs(X2 - x2) < t,
                        abs(Y1 - y1) < t,
                        abs(Y2 - y2) < t,
                    ]
                )
            ):  # touches edge
                keep_polygon.append(polygon)
                keep_scores.append(scores[idx])
                keep_classes.append(classes[idx])

        return keep_polygon, keep_scores, keep_classes



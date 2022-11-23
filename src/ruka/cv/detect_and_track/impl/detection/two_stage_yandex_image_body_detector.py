import cv2
import numpy as np
import ruka.util.distributed_fs as dfs
import time
import torch

from dataclasses import dataclass
from numpy.typing import NDArray
from ruka.cv.util import nms, AxisAlignedBBox, OrientedBBox, cut_box_and_make_square
from tf_image_processor import TFImageProcessor

from .methods.mdef_detr import make_mdef_detr_minus_language
from .one_shot_detector import OneShotDetector


@dataclass
class TwoStageYandexImageBodyDetectorConfig:
    n_box_proposals: int = 150
    nms_iou_threshold: float = 0.7
    image_to_embed_size: int = 224
    batch_size: int = 32
    candidate_expand_k: float = 1.1


class TwoStageYandexImageBodyDetector(OneShotDetector):
    def __init__(
        self,
        config: TwoStageYandexImageBodyDetectorConfig = TwoStageYandexImageBodyDetectorConfig()):

        super().__init__()

        self._rpn = make_mdef_detr_minus_language()
        self._image_body = None

        self._config = config

    def to(self, destination):
        self._rpn.to(destination)
        del self._image_body
        self._image_body = self._create_image_body(torch.device(destination))

        self._warmup()

        return self

    def train(self, mode: bool = True):
        self._rpn.train(mode)
        return self

    @torch.no_grad()
    def find(
        self,
        query_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        search_box: OrientedBBox = None) -> OrientedBBox:
        '''
        Look for the query object on an image and returns best bounding box

        Args:
            query_image (np.array): reference image of an object to look for in RGB format
            target_image (np.array): image to look for object on in RGB format
            search_box (OrientedBBox): search region on target image, not supported yet

        Returns:
            box (OrientedBBox): detected object bbox
        '''

        candidates, boxes = self._propose_boxes(target_image)
        target_embedding, candidates_embeddings = self._embed(query_image, candidates)

        best_ind = self._find_closest_embedding(target_embedding, candidates_embeddings)
        return boxes[best_ind].to_oriented()

    def _propose_boxes(self, target_image):
        boxes, scores = self._rpn(target_image)
        boxes = boxes.to_axis_aligned()

        ind = np.argsort(scores)[-self._config.n_box_proposals:]
        boxes, _ = nms(
            boxes[ind], np.array(scores)[ind],
            self._config.nms_iou_threshold)

        candidates = []
        for i in range(boxes.n_boxes):
            candidate = cut_box_and_make_square(
                target_image,
                boxes[i],
                expand_k=self._config.candidate_expand_k)[0]

            sz = self._config.image_to_embed_size
            candidate = cv2.resize(candidate, (sz, sz))
            candidates.append(candidate)

        return candidates, boxes

    def _embed(self, query_image, candidates):
        sz = self._config.image_to_embed_size
        query_image_resized = cv2.resize(query_image, (sz, sz))
        to_embed = [query_image_resized] + candidates

        embeddings = []
        for i in range(0, len(to_embed), self._config.batch_size):
            last = min(len(to_embed), i + self._config.batch_size)
            features = self._image_body.process_images(to_embed[i:last], ['prod_v12_enc_toloka_192'])
            embeddings.append(features['prod_v12_enc_toloka_192'])

        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings[0], embeddings[1:]

    def _find_closest_embedding(self, target_embedding, candidates_embeddings):
        sim = np.sum(target_embedding[None] * candidates_embeddings, axis=1)
        sim /= np.sqrt(np.sum(target_embedding ** 2))
        sim /= np.sqrt(np.sum(candidates_embeddings ** 2, axis=1))

        return np.argmax(sim)

    def _create_image_body(self, device: torch.device):
        graph_path = dfs.cached_download(
            'aux_data/cv/weights/tf_image_processor/ver12heads.gpu.fp16.graph')

        return TFImageProcessor(
            processor_version='imagebody_for_yandex_images_v12',
            graph_path=graph_path,
            cuda_device_index=device.index,
        )

    def _warmup(self):
        '''
        First batch requires significant time,
        so to this before applying on real robot.
        '''

        sz = self._config.image_to_embed_size
        target = np.zeros((sz, sz, 3), dtype=np.uint8)
        query = np.zeros((sz, sz, 3), dtype=np.uint8)

        self.find(query, target)

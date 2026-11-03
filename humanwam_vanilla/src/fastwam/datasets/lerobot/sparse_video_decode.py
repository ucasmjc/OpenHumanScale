"""Select model-consumed RGB timestamps before decode and spatial transforms."""
from dataclasses import dataclass

from torchvision.transforms import Resize
from .transforms.image import ToTensor


@dataclass
class SparseVideoQuery:
    query: object
    indices: tuple
    observation_frames: int

    def __call__(self, timestamps, episode):
        if any(len(values) != self.observation_frames for values in timestamps.values()):
            raise ValueError('Sparse video decode received an unexpected observation window')
        selected = {key: [values[i] for i in self.indices] for key, values in timestamps.items()}
        result = self.query(selected, episode)
        result['_fastwam_video_sample_indices'] = self.indices
        return result


def install_sparse_video_decode(dataset):
    processor = dataset.lerobot_dataset.processor
    if processor is None:
        raise ValueError('Sparse video decode requires an image processor')
    # Moving time selection before a stochastic or temporal transform could
    # change the training sample. Only the verified direct-resize recipe is allowed.
    for transforms in (processor.train_transforms, processor.val_transforms):
        groups = transforms.values() if isinstance(transforms, dict) else [transforms]
        for group in groups:
            if any(type(transform) not in (ToTensor, Resize) for transform in group):
                raise ValueError('Sparse video decode requires deterministic ToTensor/Resize transforms')
    for inner in dataset.lerobot_dataset.multi_dataset._datasets:
        inner._query_videos = SparseVideoQuery(
            inner._query_videos, tuple(dataset.video_sample_indices), dataset.num_frames)

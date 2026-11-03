"""Exercise the ordinary RoboTwin video route after removing pretrain readers."""
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset


def test_episode_timestamps_keep_video_offset_and_frame_order(tmp_path, monkeypatch):
    monkeypatch.setenv('FASTWAM_PYAV_DECODE_THREADS', '1')
    path = tmp_path / 'episode.mp4'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = 'yuv420p'
        for value in (0, 80, 160):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), value, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    staged = []

    def stage(path):
        staged.append(path)
        return path

    dataset = SimpleNamespace(root=tmp_path,
        meta=SimpleNamespace(get_video_file_path=lambda ep, key: 'episode.mp4',
                             get_video_timestamp_offset=lambda ep, key: 0.1),
        file_cache=SimpleNamespace(stage=stage), tolerance_s=0.02, video_backend='pyav')
    result = LeRobotDataset._query_videos(dataset, {'cam_high': [0.0, 0.1]}, 0)['cam_high']
    assert result.shape == (2, 3, 16, 16)
    assert staged == [path]
    assert torch.isfinite(result).all()
    assert result[0].mean().item() == pytest.approx(80 / 255, abs=0.03)
    assert result[1].mean().item() == pytest.approx(160 / 255, abs=0.03)


def test_pretrain_archive_binding_is_not_silently_decoded(tmp_path):
    (tmp_path / 'video_binding.json').write_text('{}')
    dataset = SimpleNamespace(root=tmp_path)
    with pytest.raises(ValueError, match='video_binding.json archive inputs'):
        LeRobotDataset._query_videos(dataset, {'cam_high': [0.0]}, 0)

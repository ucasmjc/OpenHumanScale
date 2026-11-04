"""Exercise the ordinary RoboTwin video route after removing pretrain readers."""
from types import SimpleNamespace
import json

import av
import numpy as np
import pytest
import torch

from humanwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


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


def test_existing_dataset_still_loads_without_recording_utilities(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    def no_download(*args, **kwargs):
        raise AssertionError('Local training data must not require a network request')

    monkeypatch.setattr(LeRobotDatasetMetadata, 'pull_from_repo', no_download)
    meta = tmp_path / 'meta'
    meta.mkdir()
    info = {
        'codebase_version': 'v2.1', 'robot_type': 'test', 'fps': 10,
        'total_episodes': 1, 'total_frames': 2, 'total_tasks': 1,
        'total_chunks': 1, 'chunks_size': 1000,
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path': None,
        'features': {'action': {'dtype': 'float32', 'shape': [2], 'names': None}},
    }
    (meta / 'info.json').write_text(json.dumps(info))
    (meta / 'tasks.jsonl').write_text(json.dumps({'task_index': 0, 'task': 'move object'}) + '\n')
    (meta / 'episodes.jsonl').write_text(json.dumps({'episode_index': 0, 'length': 2, 'tasks': ['move object']}) + '\n')
    data = tmp_path / 'data/chunk-000'
    data.mkdir(parents=True)
    pq.write_table(pa.table({
        'action': pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
        'episode_index': [0, 0], 'index': [0, 1], 'task_index': [0, 0],
        'frame_index': [0, 1], 'timestamp': [0.0, 0.1],
    }), data / 'episode_000000.parquet')
    metadata = LeRobotDatasetMetadata('test/local', root=tmp_path, skip_episode_stats=True)
    dataset = LeRobotDataset('test/local', root=tmp_path, metadata=metadata,
                            lazy_parquet_loading=True, video_backend='pyav')
    assert len(dataset) == 2
    torch.testing.assert_close(dataset[1]['action'], torch.tensor([1.0, 0.0]))

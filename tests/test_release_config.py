import copy
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from openhumanscale.config import load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    cfg = load_config(ROOT / 'configs/robotwin.yaml')
    # Config-only fixtures do not claim these input resources exist.
    cfg.paths = {k: f'/example/{k}' for k in cfg.paths}
    return cfg


def test_portable_main_config(config):
    report = validate_config(config)
    assert report['task_roots'] == 50
    assert report['runtime_verified'] is False
    assert config.max_steps == 24000
    assert config.model.video_vae_layout == 'robotwin_packed_multiview'
    assert config.release.profile == 'idm_w0_a2_50kh'
    assert config.data.video_augmentation.light_field.probability == 0.8
    assert config.data.video_augmentation.geometric_jitter.probability == 0.8


@pytest.mark.parametrize('key,value', [
    ('data.train.val_set_proportion', 0.01),
    ('data.train.validate_norm_stats_provenance', False),
    ('model.action_dit_config.separate_ego_action_tokenizer', False),
    ('release.profile', 'idm_head_50kh'),
    ('release.purpose', 'historical_reference'),
    ('resume_step', 24000),
    ('resume_mot_skip_prefixes', ['action_expert']),
    ('model.skip_dit_load_from_pretrain', False),
    ('data.ego_replay', {'enabled': True}),
])
def test_invalid_training_contract_rejected(config, key, value):
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError):
        validate_config(config)


def test_missing_paths_fail_before_importing_model():
    with pytest.raises(Exception, match='Missing mandatory value'):
        validate_config(load_config(ROOT / 'configs/robotwin.yaml'))


def test_duplicate_task_roots_rejected(config):
    config.data.train.dataset_dirs[1] = config.data.train.dataset_dirs[0]
    with pytest.raises(ValueError, match='distinct'):
        validate_config(config)


def test_validation_does_not_mutate_config(config):
    before = copy.deepcopy(OmegaConf.to_container(config, resolve=False))
    validate_config(config)
    assert OmegaConf.to_container(config, resolve=False) == before


def test_missing_checkpoint_rejected(config):
    with pytest.raises(FileNotFoundError):
        validate_config(config, check_files=True)


def test_current_paper_result_is_distinct_from_training_schedule(config):
    paper = json.loads((ROOT / 'release/paper_alignment.json').read_text())
    assert paper['source_profile'] == config.release.profile
    assert paper['lineage']['ego_hours'] == 50000
    assert paper['lineage']['posttrain_schedule_steps'] == config.max_steps == 24000
    assert paper['main_result']['step'] == paper['lineage']['posttrain_selected_step'] == 21000
    scores = paper['main_result']['scores']
    assert all(round(s['successes'] / s['episodes'] * 100, 2) == s['sr'] for s in scores)
    assert round(sum(s['sr'] for s in scores) / len(scores), 2) == float(paper['main_result']['mean'])

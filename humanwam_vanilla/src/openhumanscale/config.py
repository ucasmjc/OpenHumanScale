# Modified by Open-HumanScale contributors: package namespace renamed to humanwam.
"""Validate portable RoboTwin configs before importing the training stack."""
import json
from pathlib import Path

from omegaconf import OmegaConf

BASE_PROFILE = 'idm_w0_a2_50kh'


def load_config(config_path, paths_path=None, overrides=()):
    cfg = OmegaConf.load(config_path)
    if paths_path:
        cfg = OmegaConf.merge(cfg, {'paths': OmegaConf.load(paths_path)})
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def validate_config(cfg, *, check_files=False):
    """Return metadata checks; this does not claim loader or GPU validation."""
    value = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    release = value.get('release', {})
    if release.get('profile') != BASE_PROFILE:
        raise ValueError('This source draft supports only idm_w0_a2_50kh; other profiles need compatibility review')
    if release.get('purpose') != 'new_finetune':
        raise ValueError('Historical reference configs are records, not a new training recipe')
    data = value['data']
    train = data['train']
    if train.get('_target_') != 'humanwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset':
        raise ValueError('Only the RoboTwin RobotVideoDataset is supported')
    if train.get('val_set_proportion') != 0.0 or train.get('is_training_set') is not True:
        raise ValueError('New training must use the full intended corpus: val_set_proportion=0.0, is_training_set=true')
    if train.get('validate_norm_stats_provenance') is not True:
        raise ValueError('Normalization provenance validation must remain enabled')
    for key in ('ego_replay', 'edit_mix', 'wds'):
        if data.get(key):
            raise ValueError(f'data.{key} is not supported by this RoboTwin fine-tuning entry point')
    model = value['model']
    if model.get('_target_') != 'humanwam.runtime.create_fastwam' or model.get('model_class') != 'humanwam.models.wan22.fastwam_idm.FastWAMIDM':
        raise ValueError('This initial source profile requires the exported IDM model factory')
    if model['action_dit_config'].get('separate_ego_action_tokenizer') is not True:
        raise ValueError('New RoboTwin training requires separate_ego_action_tokenizer=true')
    if not value.get('resume'):
        raise ValueError('Set paths.resume to a compatible full post-trained policy checkpoint')
    if value.get('resume_step') != 0:
        raise ValueError('New fine-tuning starts a new optimization schedule with resume_step=0')
    if value.get('resume_mot_skip_prefixes') or value.get('resume_skip_proprio_encoder'):
        raise ValueError('The public fine-tuning recipe must load the full policy')
    if model.get('action_dit_pretrained_path') is not None or model.get('skip_dit_load_from_pretrain') is not True:
        raise ValueError('Load the complete released policy via resume; no private pretrain initializer is required')
    roots = train.get('dataset_dirs', [])
    if not roots or len({str(Path(p).resolve()) for p in roots}) != len(roots):
        raise ValueError('dataset_dirs must contain distinct, nonempty RoboTwin task roots')
    report = dict(profile=BASE_PROFILE, task_roots=len(roots), validation_fraction=0.0,
                  filesystem_checked=check_files, runtime_verified=False)
    if check_files:
        required_files = [value['resume'], train['pretrained_norm_stats']]
        if train.get('object_conditioned_instruction_manifest'):
            required_files.append(train['object_conditioned_instruction_manifest'])
        for path in required_files:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        required_dirs = list(roots)
        for key in ('text_embedding_cache_dir', 'merged_video_cache_dir', 'video_latent_cache_dir'):
            if train.get(key):
                required_dirs.append(train[key])
        required_dirs.extend(train.get('text_embedding_fallback_cache_dirs', []))
        for path in required_dirs:
            if not Path(path).is_dir():
                raise FileNotFoundError(path)
        info = [json.loads((Path(p) / 'meta/info.json').read_text()) for p in roots]
        report['metadata_episode_count'] = sum(int(i['total_episodes']) for i in info)
        report['episode_count_note'] = 'Metadata count only; loader selection and cache coverage require runtime verification'
    return report

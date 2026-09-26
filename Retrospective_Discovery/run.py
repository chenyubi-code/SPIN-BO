#!/usr/bin/env python3
"""Run the paper's four retrospective experiments and save numeric results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COHORTS = {'biological_multi': 'bio_multi', 'biological_single': 'bio_single',
           'random_9': 'random_9', 'random_5': 'random_5'}
METHODS = {'spin_bo': 'our_method_single_target_v1',
           'response_transfer': 'mean_only_gp_ts',
           'representation_transfer': 'encoder_only_gp_ts',
           'target_only': 'target_only_gp_ts', 'alde': 'alde_dnn_ts',
           'evolvepro': 'evolvepro_rf_topn', 'random_search': 'random_search'}
METRICS = ['normalized_best_observation', 'global_optimum_found',
           'observed_recall_at_5', 'observed_recall_at_20']


def read_tasks(cohort):
    tasks = json.loads((ROOT / 'Data/initial_designs.json').read_text())['tasks']
    return [task for task in tasks if task['cohort'] == COHORTS[cohort]]


def study_config(cohort, device=None):
    from stbo_single.config import StudyConfig
    payload = json.loads((ROOT / 'configs' / f'{cohort}.json').read_text())
    study = dict(payload['study'])
    for key in ('seeds', 'recall_ks', 'gp_amplitude_bounds',
                'gp_length_relative_bounds', 'gp_noise_bounds'):
        study[key] = tuple(study[key])
    # Preserve the actual training device, rather than resolving historical 'auto' differently.
    study['encoder_device'] = device or next(iter(payload['historical_encoder_devices']))
    return StudyConfig(**study)


def load_task(task):
    from stbo_single.data import load_dataset
    csv = ROOT / 'Data' / task['csv']
    dataset = load_dataset(csv.parent / task['filesystem_id'], task['target'],
                           csv_path=csv, sources=task['sources'])
    return dataset


def random_schedule(dataset, initial, seed, total_budget):
    """The original information-free PCG64 schedule, fixed before any target query."""
    # These hashes define the original random stream; they do not certify files.
    from stbo_single.io_utils import canonical_json_bytes, sha256_strings
    payload = {'namespace': 'full-bzip-paired-random-search-v1', 'campaign_seed': int(seed),
               'candidate_ids_sha256': sha256_strings(dataset.candidate_ids),
               'pair_ids_sha256': sha256_strings(dataset.pair_ids)}
    derived = int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:16], 'big')
    remaining_mask = dataset.feasible.copy()
    remaining_mask[initial] = False
    remaining = np.flatnonzero(remaining_mask)
    np.random.Generator(np.random.PCG64(derived)).shuffle(remaining)
    return np.concatenate((initial, remaining[:total_budget-len(initial)]))


def embedding_values(dataset, cohort, cache_root):
    group = 'biological' if cohort.startswith('biological') else 'biology_agnostic'
    directory = cache_root / 'embeddings' / group
    index_path, values_path = directory / 'sequence_index.csv', directory / 'embeddings.npy'
    if not index_path.is_file() or not values_path.is_file():
        raise FileNotFoundError(f'ESM cache missing. Run generate_embeddings.py --cohort {group}.')
    index = pd.read_csv(index_path)
    if (not {'sequence', 'embedding_id'}.issubset(index.columns)
        or not index.sequence.is_unique
        or index.embedding_id.astype(int).tolist() != list(range(len(index)))):
        raise ValueError('ESM index must contain unique sequences and ordered contiguous IDs')
    values = np.load(values_path, mmap_mode='r', allow_pickle=False)
    if values.shape != (len(index), 1280) or values.dtype != np.float32:
        raise ValueError('ESM cache must be a float32 matrix with 1280 columns')
    lookup = dict(zip(index.sequence, index.embedding_id.astype(int)))
    sequences = [*dataset.mutant_sequences, *dataset.source_partner_sequences,
                 dataset.target_partner_sequence]
    missing = set(sequences).difference(lookup)
    if missing:
        raise ValueError(f'{len(missing)} required sequences are missing from the ESM cache')
    matrix = np.asarray(values[[lookup[sequence] for sequence in sequences]])
    if not np.isfinite(matrix).all():
        raise ValueError('ESM cache contains nonfinite entries')
    n, s = dataset.candidate_count, len(dataset.sources)
    return matrix[:n], matrix[n:n+s], matrix[-1]


def build_methods(dataset, task, cohort, selected, config, cache_root):
    from stbo_single.data import build_mean_basis
    from stbo_single.encoder import prepare_encoder
    from stbo_single.gp import distance_matrix, median_pairwise_distance
    from stbo_single.proposed import ProposedMethod
    from stbo_benchmark.features import prepare_comparator_features, build_alde_onehot
    from stbo_benchmark.models import TargetOnlyGPTS, ALDEDNNTS, EvolveProRFTOPN
    from stbo_benchmark.ablations import MeanOnlyGP, EncoderOnlyGP
    built = {}
    if selected == ['random_search']:
        return built
    if set(selected).issubset({'random_search', 'alde'}):
        onehot, positions = build_alde_onehot(dataset)
        built['alde'] = ALDEDNNTS(onehot, positions, config)
        return built
    raw, partners, target = embedding_values(dataset, cohort, cache_root)
    features = prepare_comparator_features(dataset, raw)
    basis = build_mean_basis(dataset.source_outcomes, config.mean_rank_relative_tolerance)
    if set(selected) & {'spin_bo', 'representation_transfer'}:
        encoder = prepare_encoder(dataset, raw, partners, target,
                    cache_root / 'encoders' / task['task_id'], config)
        latent = encoder.target_latent
        distances = distance_matrix(latent)
        d_med = median_pairwise_distance(distances=distances)
        if 'spin_bo' in selected:
            built['spin_bo'] = ProposedMethod(basis.basis, latent, distances, d_med,
                               basis.raw_from_basis_transform, dataset.sources, config)
        if 'representation_transfer' in selected:
            built['representation_transfer'] = EncoderOnlyGP(latent, distances, d_med, config)
    if 'response_transfer' in selected:
        built['response_transfer'] = MeanOnlyGP(basis.basis, features.target_gp_scaled_esm,
             features.target_gp_distances, features.target_gp_d_med, features.target_gp_rms, config)
    if 'target_only' in selected:
        built['target_only'] = TargetOnlyGPTS(features.target_gp_scaled_esm,
             features.target_gp_distances, features.target_gp_d_med, features.target_gp_rms, config)
    if 'alde' in selected:
        built['alde'] = ALDEDNNTS(features.alde_onehot, features.alde_positions, config)
    if 'evolvepro' in selected:
        built['evolvepro'] = EvolveProRFTOPN(raw, dataset.pair_ids, config)
    return built


def campaign(dataset, task, method_name, method, config, seed):
    from stbo_single.data import make_retrospective_oracle
    initial = np.asarray(task['initial_indices'][str(seed)], dtype=np.int64)
    schedule = random_schedule(dataset, initial, seed, config.total_budget) if method_name == 'random_search' else None
    oracle = make_retrospective_oracle(dataset, config.total_budget)
    observed = initial.tolist()
    y = oracle.query(initial).tolist()
    for round_id in range(config.adaptive_rounds):
        indices = np.asarray(observed, dtype=int)
        if schedule is not None:
            selected = schedule[len(observed):len(observed)+config.batch_size]
        else:
            snapshot = method.fit_predict(indices, np.asarray(y, dtype=np.float64), seed=seed, round_id=round_id)
            decision = method.select_batch(snapshot, indices, dataset.feasible,
                            dataset.pair_ids, seed=seed, round_id=round_id,
                            batch_size=config.batch_size)
            selected = np.asarray(decision.indices, dtype=np.int64)
        if len(selected) != config.batch_size:
            raise ValueError('Acquisition did not fill the batch')
        y.extend(oracle.query(selected).tolist())
        observed.extend(selected.tolist())
    # Evaluation is the only consumer of complete target truth.
    truth = oracle.truth_for_evaluation()
    ranking = np.lexsort((dataset.pair_ids, -truth))
    maximum, minimum = float(truth.max()), float(truth.min())
    rows = []
    for round_id, budget in enumerate(config.budgets):
        selected = np.asarray(observed[:budget])
        best = float(truth[selected].max())
        rows.append({'task_id': task['task_id'], 'dataset': task['panel_id'],
          'landscape_id': dataset.landscape_id, 'target': dataset.target,
          'source': dataset.sources[0] if len(dataset.sources) == 1 else '',
          'method_id': METHODS[method_name], 'seed': seed, 'round_id': round_id,
          'budget': budget, 'best_observation': best, 'simple_regret': maximum-best,
          'normalized_best_observation': (best-minimum)/(maximum-minimum),
          'global_optimum_found': float(np.isclose(maximum-best, 0, rtol=0, atol=1e-10)),
          'observed_recall_at_5': len(np.intersect1d(selected, ranking[:5]))/5,
          'observed_recall_at_20': len(np.intersect1d(selected, ranking[:20]))/20})
    queries = pd.DataFrame({'query_order': np.arange(1,config.total_budget+1),
                'candidate_index': observed, 'pair_id': dataset.pair_ids[observed], 'observed_y': y})
    return pd.DataFrame(rows), queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True, choices=COHORTS)
    parser.add_argument('--list-tasks', action='store_true')
    parser.add_argument('--task', action='append', help='Exact task ID; repeat to select several')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--encoder-device', choices=('cpu','cuda','mps'))
    parser.add_argument('--cache-root', type=Path, default=ROOT/'Cache')
    parser.add_argument('--output', type=Path, default=ROOT/'Results')
    parser.add_argument('--force', action='store_true', help='Replace a previously completed result')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    tasks = read_tasks(args.cohort)
    if args.task:
        unknown = set(args.task).difference(task['task_id'] for task in tasks)
        if unknown:
            parser.error(f'Unknown task IDs: {sorted(unknown)}')
        tasks = [task for task in tasks if task['task_id'] in args.task]
    if args.list_tasks:
        for task in tasks:
            print(task['task_id'])
        return
    import torch
    from stbo_single.io_utils import atomic_write_csv, atomic_write_json
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    config = study_config(args.cohort, args.encoder_device)
    selected_seeds = config.seeds if args.seeds is None else args.seeds
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds) or not set(selected_seeds).issubset(config.seeds):
        parser.error('--seeds must be a nonempty unique subset of the paper seeds')
    for task in tasks:
        dataset = load_task(task)
        config.validate(dataset.candidate_count)
        if config.study_namespace != task['study_namespace']:
            raise ValueError('Task and configuration RNG namespaces disagree')
        pending=[]
        for name in args.methods:
            for seed in selected_seeds:
                destination=args.output/task['task_id']/METHODS[name]/f'seed_{seed}'
                if not args.force and (destination/'COMPLETED.json').is_file():
                    continue
                pending.append((name,seed,destination))
        if not pending:
            continue
        methods=build_methods(dataset,task,args.cohort,list(dict.fromkeys(name for name,_,_ in pending)),config,args.cache_root)
        for name,seed,destination in pending:
            print(f"RUN {task['task_id']} {name} seed={seed}",flush=True)
            (destination/'COMPLETED.json').unlink(missing_ok=True)
            metrics,queries=campaign(dataset,task,name,methods.get(name),config,seed)
            atomic_write_csv(destination/'metrics.csv',metrics)
            atomic_write_csv(destination/'queries.csv',queries)
            atomic_write_json(destination/'run.json',{'task_id':task['task_id'], 'method_id':METHODS[name],
              'seed':seed,'study':config.as_dict(),'dataset':dataset.model_metadata(),
              'execution_threads':args.threads,
              'method':{'acquisition':'uniform_random_without_replacement'} if name=='random_search' else methods[name].configuration()})
            atomic_write_json(destination/'COMPLETED.json', {'status': 'complete'})
    print('Complete. Run aggregate.py to summarize a complete cohort or a selected method.')


if __name__ == '__main__':
    main()

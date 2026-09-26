#!/usr/bin/env python3
"""Write numeric equal-panel discovery trajectories and bootstrap intervals."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from run import COHORTS, METHODS, METRICS, ROOT, read_tasks, study_config


def aggregate(cohort, result_root, output, methods):
    tasks = read_tasks(cohort)
    config = study_config(cohort)
    frames=[]
    for task in tasks:
        for method in methods:
            for seed in config.seeds:
                directory=result_root/task['task_id']/METHODS[method]/f'seed_{seed}'
                marker=directory/'COMPLETED.json'
                if not marker.is_file():
                    raise RuntimeError(f"Incomplete cohort: missing {task['task_id']}/{method}/seed_{seed}")
                frame=pd.read_csv(directory/'metrics.csv')
                if frame.budget.tolist()!=list(config.budgets) or not frame.seed.eq(seed).all() or not frame.method_id.eq(METHODS[method]).all():
                    raise ValueError('Invalid campaign budget, seed or method')
                if not frame.task_id.eq(task['task_id']).all() or not frame.dataset.eq(task['panel_id']).all():
                    raise ValueError('Result belongs to a different target problem')
                frames.append(frame)
    data=pd.concat(frames,ignore_index=True)
    if not np.isfinite(data[METRICS]).all().all():
        raise ValueError('Nonfinite discovery metric')
    problems=data.groupby(['task_id','dataset','landscape_id','method_id','budget'],as_index=False)[METRICS].mean()
    panels=problems.groupby(['dataset','landscape_id','method_id','budget'],as_index=False)[METRICS].mean()
    means=panels.groupby(['method_id','budget'],as_index=False)[METRICS].mean()
    panel_ids=sorted(panels.dataset.unique())
    backgrounds=sorted(panels.landscape_id.unique())
    draws=np.random.default_rng(20260921).integers(0,len(backgrounds),size=(2500,len(backgrounds)))
    panel_backgrounds=panels.drop_duplicates('dataset').set_index('dataset').loc[panel_ids,'landscape_id']
    weights=np.stack([(draws==backgrounds.index(background)).sum(axis=1) for background in panel_backgrounds],axis=1)
    bands=[]
    for method in methods:
        for metric in METRICS:
            values=panels[panels.method_id.eq(METHODS[method])].pivot(index='dataset',columns='budget',values=metric).loc[panel_ids,list(config.budgets)].to_numpy()
            boot=weights@values/weights.sum(axis=1)[:,None]
            low,high=np.quantile(boot,[.025,.975],axis=0)
            for budget,lo,hi in zip(config.budgets,low,high):
                bands.append({'method_id':METHODS[method],'metric':metric,'budget':budget,'ci95_low':lo,'ci95_high':hi})
    output.mkdir(parents=True,exist_ok=True)
    for name,frame in [('seed_metrics',data),('problem_curves',problems),('panel_curves',panels),('curves',means),('bootstrap_intervals',pd.DataFrame(bands))]:
        frame.to_csv(output/f'{cohort}_{name}.csv',index=False)
    trajectories=[]
    for method,frame in means.groupby('method_id'):
        frame=frame.sort_values('budget')
        trajectories.append({'method_id':method,**{metric:float(np.trapezoid(frame[metric],frame.budget)/180) for metric in METRICS}})
    pd.DataFrame(trajectories).to_csv(output/f'{cohort}_trajectory_averages.csv',index=False)
    print(means[means.budget.isin([100,200])].to_string(index=False))
    return means


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',required=True,choices=COHORTS)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--results',type=Path,default=ROOT/'Results')
    parser.add_argument('--output',type=Path,default=ROOT/'Results/summary')
    args=parser.parse_args()
    aggregate(args.cohort,args.results,args.output,args.methods)


if __name__=='__main__':
    main()

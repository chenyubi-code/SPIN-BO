#!/usr/bin/env python3
"""Generate the paper's sequence-keyed ESM-2 features without distributing weights or vectors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def generate(generation, specification, checkpoint, output, device, dtype):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
    frame=pd.read_csv(ROOT/'Data'/generation['sequence_index'])
    if not len(frame) or not frame.sequence.is_unique:
        raise ValueError('Invalid sequence manifest')
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    cls=AutoModelForMaskedLM if generation['loader'].startswith('AutoModelForMaskedLM') else AutoModel
    model_dtype=getattr(torch,dtype)
    kwargs={'dtype':model_dtype} if int(transformers.__version__.split('.')[0])>=5 else {'torch_dtype':model_dtype}
    model=cls.from_pretrained(checkpoint,local_files_only=True,**kwargs)
    model.eval().to(device)
    backbone=model.esm if generation['loader'].startswith('AutoModelForMaskedLM') else model
    matrix=np.empty((len(frame),specification['hidden_size']),dtype=np.float32)
    for batch_id,rows in frame.groupby('original_batch_id',sort=True):
        sizes=rows.original_batch_size.unique()
        if len(sizes)!=1:
            raise ValueError('Inconsistent historical batch size')
        # Every sequence has 62 residues. Fill omitted slots with an included sequence
        # to preserve tensor shape and retained positions without adding unrelated data.
        sequences=[rows.iloc[0].sequence]*int(sizes[0])
        for row in rows.itertuples():
            sequences[int(row.original_batch_position)]=row.sequence
        encoded=tokenizer(sequences,add_special_tokens=True,padding=True,
           return_attention_mask=True,return_special_tokens_mask=True,return_tensors='pt')
        special=encoded.pop('special_tokens_mask').bool()
        valid=encoded['attention_mask'].bool() & ~special
        if not np.array_equal(valid.sum(dim=1).numpy(),np.asarray(list(map(len,sequences)))):
            raise ValueError('Tokenizer residue count mismatch')
        with torch.inference_mode():
            hidden=backbone(**{key:value.to(device) for key,value in encoded.items()},return_dict=True).last_hidden_state.float()
            mask=valid.to(device).unsqueeze(-1)
            pooled=((hidden*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)).cpu().numpy().astype(np.float32)
        for row in rows.itertuples():
            matrix[row.Index]=pooled[int(row.original_batch_position)]
        if int(batch_id)%100==0:
            print(f"{generation['id']}: historical batch {batch_id}",flush=True)
    if not np.isfinite(matrix).all():
        raise RuntimeError('Nonfinite ESM features')
    output.mkdir(parents=True,exist_ok=True)
    np.save(output/'embeddings.npy',matrix,allow_pickle=False)
    pd.DataFrame({'embedding_id':np.arange(len(frame)),'sequence':frame.sequence}).to_csv(output/'sequence_index.csv',index=False)
    settings={'generation_set':generation['id'],'device':device,'inference_dtype':dtype,
              'pooling_dtype':'float32','embedding_dtype':'float32','status':'complete',
              'packages':{'torch':torch.__version__,'transformers':transformers.__version__},
              'sequences':len(frame)}
    (output/'metadata.json').write_text(json.dumps(settings,indent=2)+'\n')
    print(f"Generated {generation['id']}: {len(frame)} sequence features.")


def assemble(group,specification,cache_root):
    ids=['biological'] if group=='biological' else ['biology_agnostic','biology_agnostic_atf4']
    parts=[];matrices=[]
    for name in ids:
        directory=cache_root/'embedding_sets'/name
        if not (directory/'embeddings.npy').is_file() or not (directory/'sequence_index.csv').is_file():
            print(f'Missing generation set {name}; generate it before assembling {group}.')
            return
        part=pd.read_csv(directory/'sequence_index.csv')
        values=np.load(directory/'embeddings.npy',allow_pickle=False)
        if values.shape!=(len(part),specification['hidden_size']) or values.dtype!=np.float32:
            raise ValueError(f'Embedding array dimensions or dtype do not match the {name} sequence index')
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite embeddings in {name}')
        parts.append(part);matrices.append(values)
    frame=pd.concat(parts,ignore_index=True);values=np.vstack(matrices)
    if not frame.sequence.is_unique:
        raise ValueError('Generation sets contain overlapping sequences')
    frame.embedding_id=np.arange(len(frame))
    output=cache_root/'embeddings'/group;output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'sequence_index.csv',index=False)
    np.save(output/'embeddings.npy',values,allow_pickle=False)
    settings={'group':group,'generation_sets':ids,'sequences':len(frame),
              'embedding_dtype':'float32','status':'complete'}
    (output/'metadata.json').write_text(json.dumps(settings,indent=2)+'\n')
    print(f'Assembled {group}: {len(frame)} sequence features.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',required=True,choices=('biological','biology_agnostic'))
    parser.add_argument('--generation-set',choices=('biological','biology_agnostic','biology_agnostic_atf4'),help='Generate one set, permitting its original package environment')
    parser.add_argument('--checkpoint',type=Path,help='Local Hugging Face checkpoint directory; otherwise download the published model')
    parser.add_argument('--cache-root',type=Path,default=ROOT/'Cache')
    parser.add_argument('--device',choices=('cpu','cuda','mps'),default='mps')
    parser.add_argument('--dtype',choices=('float16','float32','bfloat16'),default='float16')
    parser.add_argument('--assemble-only',action='store_true')
    args=parser.parse_args()
    specification=json.loads((ROOT/'Data/embedding_metadata/specification.json').read_text())
    ids=['biological'] if args.cohort=='biological' else ['biology_agnostic','biology_agnostic_atf4']
    if args.generation_set and args.generation_set not in ids:
        parser.error('--generation-set does not belong to the selected cohort')
    if not args.assemble_only:
        checkpoint=args.checkpoint
        if checkpoint is None:
            from huggingface_hub import snapshot_download
            checkpoint=Path(snapshot_download(specification['model_id'],
               local_dir=args.cache_root/'checkpoints/esm2_t33_650M_UR50D',
               allow_patterns=specification['checkpoint_files']))
        for generation in specification['generation_sets']:
            if generation['id'] not in ([args.generation_set] if args.generation_set else ids):
                continue
            generate(generation,specification,checkpoint,args.cache_root/'embedding_sets'/generation['id'],
                     args.device,args.dtype)
    assemble(args.cohort,specification,args.cache_root)


if __name__=='__main__':
    main()

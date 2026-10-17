"""Export authentic encoder features for one registered H&E/ST patch.

Run inside the respective upstream encoder environment. See docs/usage.md.
No model weights or additional datasets are bundled by this script.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def save_array(path, array):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(array).all():
        raise ValueError('Encoder produced nonfinite features')
    np.save(path, np.asarray(array, dtype=np.float32))


def gigapath(args):
    import timm
    import torch
    from PIL import Image
    from torchvision import transforms
    image = np.load(args.he) if args.he.suffix == '.npy' else np.asarray(Image.open(args.he).convert('RGB'))
    if image.shape != (5120, 5120, 3):
        raise ValueError('Expected registered native RGB H&E, shape [5120,5120,3] at 0.5 um/pixel')
    image = np.clip(image, 0, 255).astype(np.uint8)
    # The first vector describes the tissue tile; the following vectors describe
    # the row-major 16 x 16 cellular grid (each 320 px / 160 um natively).
    tiles = [image] + [image[r:r + 320, c:c + 320] for r in range(0, 5120, 320) for c in range(0, 5120, 320)]
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    model = timm.create_model(args.model, pretrained=True).to(args.device).eval()
    features = []
    with torch.inference_mode():
        for start in range(0, len(tiles), args.batch_size):
            batch = torch.stack([transform(Image.fromarray(tile)) for tile in tiles[start:start + args.batch_size]])
            features.append(model(batch.to(args.device)).float().cpu().numpy())
    output = np.concatenate(features)
    if output.shape != (257, 1536):
        raise ValueError(f'Expected 257 original Prov-GigaPath tile vectors of width 1536, got {output.shape}')
    save_array(args.output, output)


def scgpt(args):
    import scanpy as sc
    from anndata import AnnData
    from scipy import sparse
    from scgpt_spatial.tasks import embed_data
    from scgpt_spatial.tasks import cell_emb as upstream_embedding
    names = np.loadtxt(args.gene_names, dtype=str, ndmin=1)
    order = np.load(args.gene_order).astype(int)
    if len(names) != len(order):
        raise ValueError('Gene names must correspond to gene_order entries')
    stop = args.gene_start + args.gene_num
    indices = order[args.gene_start:stop]
    selected = names[args.gene_start:stop]
    if len(selected) != args.gene_num:
        raise ValueError('Selected gene group is incomplete')
    matrix = sparse.load_npz(args.spot)[:, indices].toarray()
    if matrix.shape != (26 * 26, args.gene_num):
        raise ValueError('Expected 676 row-major ST grid points')
    nonzero = matrix.sum(1) > 0
    if not nonzero.any():
        raise ValueError('The selected ST group has no nonzero spots')
    adata = AnnData(matrix[nonzero].astype(np.float32))
    encoder_names = selected.tolist()
    if args.ortholog_map is not None:
        with args.ortholog_map.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        mapping = {row['source_gene']: row['encoder_gene'] for row in rows}
        missing = [gene for gene in encoder_names if gene not in mapping]
        if missing:
            raise ValueError(f'Missing explicit encoder orthologs: {missing}')
        encoder_names = [mapping[gene] for gene in encoder_names]
    if len(set(encoder_names)) != len(encoder_names):
        raise ValueError('Encoder gene names must be unique after mapping')
    import json
    vocab = json.loads((args.model_dir / 'vocab.json').read_text(encoding='utf-8'))
    missing = [gene for gene in encoder_names if gene not in vocab]
    if missing:
        raise ValueError(f'Genes absent from scGPT-spatial vocabulary: {missing}; supply a verified ortholog map')
    adata.var_names = encoder_names
    # Upstream embed_data performs slide-mean scaling, gene-mean scaling and
    # token binning; do not apply another log1p or normalization here.
    original_load = upstream_embedding.load_pretrained
    load_audit = {}

    def checked_load(model, pretrained_params, **kwargs):
        if not model.use_fast_transformer:
            pretrained_params = {key.replace('Wqkv.', 'in_proj_'): value
                                 for key, value in pretrained_params.items()}
        state = model.state_dict()
        required = [key for key in state if key.startswith(
            ('encoder.', 'value_encoder.', 'transformer_encoder.'))]
        unmatched = [key for key in required if key not in pretrained_params or
                     pretrained_params[key].shape != state[key].shape]
        if unmatched:
            raise ValueError(f'scGPT-spatial encoder checkpoint mismatch: {unmatched[:8]}')
        load_audit.update(required_tensors=len(required),
                          required_parameters=sum(state[key].numel() for key in required),
                          matched_tensors=len(required),
                          checkpoint_sha256=hashlib.sha256((args.model_dir / 'best_model.pt').read_bytes()).hexdigest())
        return original_load(model, pretrained_params, **kwargs)

    upstream_embedding.load_pretrained = checked_load
    embedded = embed_data(adata, args.model_dir, gene_col='index', device=args.device,
                          batch_size=args.batch_size, use_fast_transformer=False)
    if not load_audit:
        raise RuntimeError('Upstream scGPT-spatial checkpoint loader was not called')
    values = np.asarray(embedded.obsm['X_scGPT'])
    full = np.zeros((676, values.shape[1]), dtype=np.float32)
    full[nonzero] = values
    save_array(args.output, full.T.reshape(values.shape[1], 26, 26))
    np.savetxt(args.output.with_suffix('.genes.txt'), selected, fmt='%s')
    args.encoder_genes = encoder_names
    args.checkpoint_audit = load_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='encoder', required=True)
    he = sub.add_parser('gigapath')
    he.add_argument('--he', type=Path, required=True)
    he.add_argument('--model', default='hf_hub:prov-gigapath/prov-gigapath')
    st = sub.add_parser('scgpt')
    st.add_argument('--spot', type=Path, required=True)
    st.add_argument('--gene-order', type=Path, required=True)
    st.add_argument('--gene-names', type=Path, required=True)
    st.add_argument('--model-dir', type=Path, required=True)
    st.add_argument('--ortholog-map', type=Path,
                    help='CSV source_gene,encoder_gene; required when source symbols are absent from the encoder vocabulary')
    st.add_argument('--gene-start', type=int, default=0)
    st.add_argument('--gene-num', type=int, default=5)
    for p in (he, st):
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--device', default='cuda')
        p.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.encoder == 'gigapath':
        gigapath(args)
    else:
        scgpt(args)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata['output_sha256'] = digest
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved {args.output}, SHA256 {digest}')


if __name__ == '__main__':
    main()

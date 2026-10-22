"""Persistence/reader contracts: tile ownership, crash recovery and lazy sampling.

No fitting is needed: schema-correct deterministic outcomes distinguish storage
and indexing failures from the EM mathematics owned by lower-level tests.
"""
import hashlib
import json

import numpy as np
import pytest
from filelock import Timeout

from surface_computation import simulation_corpus as storage
from surrogate_training.wnm import data


@pytest.fixture
def example():
    design = np.array([[10,20,30,60],[20,10,30,90],[30,40,30,60],
        [40,30,30,90],[50,60,30,60]],np.float32)
    rows,sims = len(design),7
    arrays = {name:np.zeros(storage.array_shape(name,rows,sims),dtype)
              for name,dtype in storage.DTYPES.items()}
    arrays['bias'][:] = np.arange(rows*sims*2).reshape(rows,sims,2)-30
    arrays['em_parameters'][:] = np.arange(rows*sims*10).reshape(rows,sims,5,2)
    arrays['em_converged'][:] = True
    arrays['em_any_best_converged'][:] = True
    arrays['em_iterations'][:] = np.arange(rows*sims).reshape(rows,sims)
    # One masked failed winner, one repaired fit and counted losing-start failures.
    arrays['em_masked'][1,2] = True
    arrays['bias'][1,2] = np.nan
    arrays['em_converged'][1,2] = False
    arrays['em_log_likelihood'][1,2] = np.nan
    arrays['em_relative_change'][1,2] = np.inf
    arrays['em_repaired'][2,0] = True
    arrays['em_material_decreases'][0,0] = 2
    arrays['em_nonfinite_starts'][0,1] = 1
    arrays['em_worst_change'][0,1] = np.nan  # all-start aggregate of that losing failure
    strata = ['a','a','b','b','c']
    signature = dict(storage_format=storage.FORMAT,n_rows=rows,n_simulations=sims,
        shard_rows=2,simulation_chunk=3,n_samples=4,seed=19,
        design_sha256=hashlib.sha256(design.tobytes()).hexdigest(),
        strata_sha256=hashlib.sha256(json.dumps(strata).encode()).hexdigest())
    return design,strata,signature,arrays


def fill(corpus,arrays):
    for a in range(0,corpus.rows,corpus.signature['shard_rows']):
        for c in range(0,corpus.simulations,corpus.signature['simulation_chunk']):
            coordinates = corpus.coordinates(a,c)
            _,b,_,d = coordinates
            corpus.write_tile(coordinates,{k:v[a:b,c:d] for k,v in arrays.items()})


@pytest.mark.parametrize('compress',[False,True])
def test_complete_corpus_export_and_sampling_match_in_memory(tmp_path,example,compress):
    design,strata,signature,arrays = example
    root = tmp_path/'corpus'
    with storage.writer(root,signature,design,strata) as corpus:
        fill(corpus,arrays)
        corpus.finish({'n_samples':4})
    with storage.Corpus(root) as corpus:
        corpus.verify()
        assert corpus.meta['em_summary']['maximum_iterations']==34
        summary = corpus.meta['em_summary']
        assert (summary['masked_fits'],summary['repaired_fits'],corpus.meta['n_masked'])==(1,1,1)
        assert (summary['material_decreases'],summary['nonfinite_starts'])==(2,1)
        for name,value in arrays.items():
            assert isinstance(corpus.arrays[name],np.memmap)
            np.testing.assert_array_equal(corpus.arrays[name],value)
    # Completion metadata must agree with the indexed tile summaries.
    original = (root/'complete.json').read_text()
    marker = json.loads(original)
    marker['meta']['n_masked'] = 0
    (root/'complete.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError,match='disagrees with its tile summaries'):
        storage.Corpus(root)
    (root/'complete.json').write_text(original)
    destination = tmp_path/'export.npz'
    storage.export_npz(root,destination,compress=compress)
    with np.load(destination) as blob:
        for name,value in arrays.items():
            assert blob[name].dtype==value.dtype and blob[name].tobytes()==value.tobytes()
        assert blob['strata'].tolist()==strata
        assert json.loads(str(blob['meta']))['generation_signature']==signature
    lazy,meta = data.load_source(str(root))
    memory = data.SampleStore(design,arrays['bias'])
    assert isinstance(lazy.bias.base,np.memmap)
    assert lazy.n_observations==memory.n_observations
    assert meta['strata']==strata
    for method,args in [('batch',(31,)),('grouped_batch',(13,5))]:
        for subset in [slice(None),[4,0,2],np.array([True,False,True,False,True])]:
            actual = getattr(lazy.subset(subset),method)(np.random.default_rng(7),*args)
            expected = getattr(memory.subset(subset),method)(np.random.default_rng(7),*args)
            for a,b in zip(actual,expected):
                np.testing.assert_array_equal(a,b)
    train,val = data.split_by_params(lazy,seed=7)
    assert train.bias.base is lazy.bias.base and val.bias.base is lazy.bias.base
    assert not (set(map(tuple,train.canonical_param_key())) & set(map(tuple,val.canonical_param_key())))


def test_interrupted_tile_corruption_and_resume(tmp_path,example,monkeypatch):
    design,strata,signature,arrays = example
    root = tmp_path/'corpus'
    with storage.writer(root,signature,design,strata) as corpus:
        first = corpus.coordinates(0,0)
        corpus.write_tile(first,{k:v[:2,:3] for k,v in arrays.items()})
        corpus.arrays['bias'][2:4,:3] = 99  # interrupted next write; no checkpoint
        corpus.arrays['bias'].flush()
        with pytest.raises(ValueError,match='coverage'):
            corpus.finish({})
        with pytest.raises(Timeout):
            with storage.writer(root,signature,design,strata):
                pass
    with pytest.raises(ValueError,match='incomplete'):
        data.load_source(str(root))
    with storage.writer(root,signature,design,strata) as corpus:
        assert corpus.valid_tile(first)
        assert not corpus.valid_tile(corpus.coordinates(2,0))
        fill(corpus,arrays)
        corpus.finish({})
        def interrupted(value):
            raise OSError('interrupted array flush')
        with monkeypatch.context() as patch:
            patch.setattr(np.memmap,'flush',interrupted)
            with pytest.raises(OSError,match='interrupted'):
                corpus.write_tile(first,{k:v[:2,:3] for k,v in arrays.items()})
        assert not (root/'complete.json').exists()
        assert not corpus.valid_tile(first)
        corpus.write_tile(first,{k:v[:2,:3] for k,v in arrays.items()})
        corpus.finish({})
        corpus.arrays['bias'][0,0,0] += 1
        corpus.arrays['bias'].flush()
        with pytest.raises(ValueError,match='checksum'):
            corpus.verify()
        assert not corpus.valid_tile(first)
        corpus.write_tile(first,{k:v[:2,:3] for k,v in arrays.items()})
        corpus.finish({})
    with storage.Corpus(root) as corpus:
        corpus.verify()
    with pytest.raises(ValueError,match='manifest does not match'):
        with storage.writer(root,dict(signature,seed=20),design,strata):
            pass
    (root/'manifest.json').unlink()
    with pytest.raises(ValueError,match='no manifest'):
        with storage.writer(root,signature,design,strata):
            pass


def test_invalid_or_incomplete_tile_cannot_publish(tmp_path,example):
    design,strata,signature,arrays = example
    with storage.writer(tmp_path/'corpus',signature,design,strata) as corpus:
        part = {k:v[:2,:3].copy() for k,v in arrays.items()}
        part['em_converged'][0,0] = False
        with pytest.raises(ValueError,match='EM validation'):
            corpus.write_tile(corpus.coordinates(0,0),part)
        part = {k:v[:2,:3].copy() for k,v in arrays.items()}
        part['bias'][0,0,0] = np.nan
        with pytest.raises(ValueError,match='nonfinite corpus tile array bias'):
            corpus.write_tile(corpus.coordinates(0,0),part)
        part = {k:v[:2,:3].copy() for k,v in arrays.items()}
        part['bias'][1,2] = 0.
        with pytest.raises(ValueError,match='inconsistent masked'):
            corpus.write_tile(corpus.coordinates(0,0),part)
        part = {k:v[:2,:3].copy() for k,v in arrays.items()}
        part['em_converged'][1,2] = True  # masked although its diagnostics say success
        with pytest.raises(ValueError,match='inconsistent masked'):
            corpus.write_tile(corpus.coordinates(0,0),part)
        with pytest.raises(ValueError,match='coverage'):
            corpus.finish({})


def test_failed_export_preserves_prior_output(tmp_path,example,monkeypatch):
    design,strata,signature,arrays = example
    root = tmp_path/'corpus'
    with storage.writer(root,signature,design,strata) as corpus:
        fill(corpus,arrays)
        corpus.finish({})
    destination = tmp_path/'export.npz'
    destination.write_bytes(b'prior complete output')
    def interrupted(*args,**kwargs):
        raise OSError('interrupted export')
    with monkeypatch.context() as patch:
        patch.setattr(np.lib.format,'write_array_header_2_0',interrupted)
        with pytest.raises(OSError,match='interrupted'):
            storage.export_npz(root,destination)
    assert destination.read_bytes()==b'prior complete output'
    storage.export_npz(root,destination)
    with np.load(destination) as exported:
        np.testing.assert_array_equal(exported['bias'],arrays['bias'])


def test_legacy_nonfinite_samples_keep_weights_without_full_mask():
    bias = np.arange(24,dtype=np.float32).reshape(3,4,2)
    bias[0,1,0] = np.nan
    store = data.SampleStore(np.ones((3,4)),bias)
    assert store.n_observations==23
    _,values,weights = store.grouped_batch(np.random.default_rng(4),100,5)
    assert np.isfinite(values).all() and (weights==0).any()


def binned(bias, bins):
    """Independent NumPy reference for half-open circular bins from -180 deg."""
    finite = np.isfinite(bias)
    index = np.floor(np.mod(np.where(finite, bias, 0.)+180., 360.)*bins/360.).astype(int) % bins
    out = np.zeros(bias.shape[:1]+(2, bins), np.int32)
    for row in range(bias.shape[0]):
        for item in (0, 1):
            np.add.at(out[row, item], index[row, finite[row, :, item], item], 1)
    return out


def test_histogram_corpus_round_trip_summaries_and_validation(tmp_path,example):
    design,strata,signature,arrays = example
    signature = dict(signature,storage_format=storage.HISTOGRAM_FORMAT,histogram_bins=8)
    raw_summary = storage.summary(arrays)
    root = tmp_path/'histograms'
    with storage.writer(root,signature,design,strata) as corpus:
        for a in range(0,corpus.rows,signature['shard_rows']):
            for c in range(0,corpus.simulations,signature['simulation_chunk']):
                coordinates = corpus.coordinates(a,c)
                _,b,_,d = coordinates
                tile = {k:v[a:b,c:d] for k,v in arrays.items()}
                counts = np.stack([np.sum(v,axis=1,dtype=np.int64) for v in (
                    ~tile['em_converged'],tile['em_capped_starts'],tile['em_nonfinite_starts'],
                    tile['em_material_decreases'],tile['em_continued_winner'],
                    tile['em_repaired'],tile['em_masked'])],axis=-1)
                part = dict(histogram=binned(tile['bias'],8)[:,None],counts=counts[:,None],
                            max_iterations=tile['em_iterations'].max(axis=1)[:,None].astype(np.int32))
                if (a,c)==(0,0):
                    bad = dict(part,histogram=part['histogram']+1)
                    with pytest.raises(ValueError,match='do not equal its trials'):
                        corpus.write_tile(coordinates,bad)
                    with pytest.raises(ValueError,match='do not match the layout'):
                        corpus.write_tile(coordinates,dict(part,bias=tile['bias']))
                corpus.write_tile(coordinates,part)
        corpus.finish({})
    with storage.Corpus(root) as corpus:
        corpus.verify()
        np.testing.assert_array_equal(corpus.histograms(),binned(arrays['bias'],8))
        # Same totals a raw corpus of these trials would record.
        assert corpus.meta['em_summary']==raw_summary and corpus.meta['n_masked']==1

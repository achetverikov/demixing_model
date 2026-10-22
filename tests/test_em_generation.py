"""Maintained generation contracts: valid diagnostics, resume, and output isolation."""
import importlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surface_computation import generate_wnm_training_data as generator
from surface_computation import likelihood_em as em
from surface_computation import em_identity


def test_kernel_and_shared_density_changes_invalidate_generation_digest(tmp_path,monkeypatch):
    root=Path(generator.__file__).resolve().parents[1]
    paths=['surface_computation/'+name for name in ['generate_wnm_training_data.py','wnm_design.py',
        'wnm_simulation.py','simulation_corpus.py','jax_fit_main.py','jax_fit_functions.py','likelihood_em.py','em_batch.py','em_identity.py',
        'cuda_em.py', 'structured_starts.py', *generator.cuda_em.SOURCE_NAMES]]+['shared/wnm.py']
    for relative in paths:
        dest=tmp_path/relative;dest.parent.mkdir(exist_ok=True,parents=True)
        dest.write_bytes((root/relative).read_bytes())
    monkeypatch.setattr(generator,'__file__',str(tmp_path/'surface_computation/generate_wnm_training_data.py'))
    original=generator._code_digest()
    kernel=tmp_path/'surface_computation/likelihood_em.py'
    kernel.write_text(kernel.read_text()+'\n# changed estimator\n')
    assert generator._code_digest()!=original
    prior=generator._code_digest()
    density=tmp_path/'shared/wnm.py';density.write_text(density.read_text()+'\n# changed density\n')
    assert generator._code_digest()!=prior
    prior=generator._code_digest()
    cuda=tmp_path/'surface_computation/cuda/likelihood_em.cuh'
    cuda.write_text(cuda.read_text()+'\n// changed CUDA kernel\n')
    assert generator._code_digest()!=prior


def test_cloud_reader_resolves_the_same_namespace_as_the_writer(monkeypatch):
    from surface_computation.object_store import ObjectStoreConfig
    status=importlib.import_module('cloud.cloud_status')
    monkeypatch.setattr(status,'load_env',lambda:None)
    monkeypatch.setattr(status.ObjectStoreConfig,'from_env',lambda:ObjectStoreConfig(bucket='example',prefix='base'))
    observed={}
    class Store:
        def __init__(self,config):observed['prefix']=config.prefix
        def list_bundle_objects(self):return []
        def list_completed_surface_ids(self):return {'surface'}
    monkeypatch.setattr(status,'SurfaceObjectStore',Store)
    monkeypatch.setenv('REDIS_HOST','example')
    def completed(registry):
        observed['registry']=registry
        return {'surface'}
    monkeypatch.setattr(status,'_redis_completed',completed)
    monkeypatch.setattr(sys,'argv',['status','--registry','completed','--n-simulations','10000','--n-samples','20'])
    monkeypatch.delenv('SIMULATION_NAMESPACE',raising=False)
    assert status.main()==0
    namespace=em_identity.grid_namespace(10000,20)
    assert observed==dict(prefix=em_identity.qualify_prefix('base',namespace),
                         registry=em_identity.qualify_name('completed',namespace))


@pytest.mark.gpu
@pytest.mark.parametrize('backend', ['jax', 'cuda'])
def test_generator_diagnostics_resume_and_stale_estimator_refusal(tmp_path,monkeypatch,backend):
    design=tmp_path/'design.npz';out=tmp_path/('generated.npz' if backend=='jax' else 'generated')
    points=np.array([[10.,10.,10.,60.],[20.,10.,10.,100.]],np.float32)
    np.savez(design,design=points,strata=['curve_a','curve_a'])
    args=['generator','--design-file',str(design),'--out',str(out),'--n-samples','4',
          '--n-simulations','2','--simulation-chunk','2','--shard-rows','2','--block-rows','2',
          '--em-backend',backend,'--resume','--crn-within-trajectories']
    if backend == 'jax':
        args += ['--em-stages','5','50','5000']
    monkeypatch.setattr(sys,'argv',args)
    generator.main()
    from surface_computation import simulation_corpus
    with (np.load(out) if backend=='jax' else simulation_corpus.Corpus(out)) as saved:
        blob = saved if backend=='jax' else dict(saved.arrays,meta=json.dumps(saved.meta))
        assert blob['bias'].shape==(2,2,2)
        assert blob['em_converged'].all() and not blob['em_nonfinite_starts'].any()
        assert blob['em_iterations'].shape==(2,2)
        assert blob['em_parameters'].shape==(2,2,5,2)
        assert json.loads(str(blob['meta']))['estimator']==em.ESTIMATOR_ID
        signature=json.loads(str(blob['meta']))['generation_signature']
        assert signature['simulation_code_sha256']==generator._code_digest()
        assert (signature['em_backend_build'] is not None)==(backend=='cuda')
    def unexpected(*args,**kwargs):
        raise AssertionError('completed shards must resume without another fit')
    monkeypatch.setattr(generator.wnm_simulation,'simulate',unexpected)
    generator.main()
    # Labels change the CRN groups even when all numeric design rows stay fixed.
    np.savez(design,design=points,strata=['curve_a','curve_b'])
    with pytest.raises(ValueError,match='manifest does not match'):
        generator.main()
    np.savez(design,design=points,strata=['curve_a','curve_a'])
    manifest=(out.with_suffix('.npz.shards') if backend=='jax' else out)/'manifest.json'
    bad=json.loads(manifest.read_text());bad['estimator']='different_estimator'
    assert bad['block_rows']==2
    assert bad['em_backend']==backend
    manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='manifest does not match'):
        generator.main()
    manifest.unlink()
    with pytest.raises(ValueError,match='existing shards have no manifest'):
        generator.main()


@pytest.mark.gpu
@pytest.mark.parametrize('full',[False,True])
def test_raw_grid_preserves_nondivisible_trial_counts_and_checks_diagnostics(tmp_path,monkeypatch,full,request):
    # Routing test: use a small analytic simulator stub, rather than refitting.
    sc=Path(generator.__file__).resolve().parent
    monkeypatch.syspath_prepend(str(sc))
    grid=importlib.import_module('surface_computation.simulated_samples_grid')
    grid._scan_entry.clear_cache()
    request.addfinalizer(grid._scan_entry.clear_cache)
    computer=grid.ChunkedGridComputer.__new__(grid.ChunkedGridComputer)
    computer.save_full_results=full;computer.fix_weights=False
    computer.algorithm='EM';computer.diagonal_covariance=True
    def simulate(key,*values,return_full_results=False,return_diagnostics=False,**kwargs):
        n=kwargs['n_simulations']
        bias=jnp.zeros((n,2))
        diagnostic=dict(converged=jnp.ones(n,dtype=bool),nonfinite_starts=jnp.zeros(n,dtype=int),
                        material_decreases=jnp.zeros(n,dtype=int),winner_decreases=jnp.zeros(n,dtype=int),
                        iterations=jnp.ones(n,dtype=int))
        result=(bias,bias,jnp.zeros((n,2,23))) if return_full_results else (bias,bias)
        return result+(diagnostic,)
    monkeypatch.setattr(grid.jfm,'simulate_dual_component_bias_distribution',simulate)
    result=computer._compute_samples_for_entry(10,20,30,None,
        dict(n_simulations=11,n_samples=4,random_seed=17),jnp.array([30.,60.]))
    assert result[0].shape==(2,11,2) and result[-1]['iterations'].shape==(2,11)
    assert (result[2] is not None)==full
    assert grid.estimator_path(grid.estimator_path(tmp_path/'samples'))==grid.estimator_path(tmp_path/'samples')
    assert em.ESTIMATOR_ID in grid.build_samples_folder_name(1000,100,'EM',True,False)
    monkeypatch.setattr(grid.config,'samples_folder',tmp_path/grid.build_samples_folder_name(1000,100,'EM',True,False))
    first=grid.run_path(tmp_path/'explicit_surfaces')
    monkeypatch.setattr(grid.config,'samples_folder',tmp_path/grid.build_samples_folder_name(1000,500,'EM',True,False))
    second=grid.run_path(tmp_path/'explicit_surfaces')
    assert first!=second and grid.run_path(second)==second

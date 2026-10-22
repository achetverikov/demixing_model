"""Memory-mapped raw outcomes with checksummed, run-bound tile checkpoints.

NumPy files are the durable corpus, not an intermediate for full-RAM assembly.
SQLite transactions publish completed tiles after their array writes flush.
The completion marker covers the exact, disjoint logical tile index. One writer
owns a corpus; readers accept only completed corpora. No JAX dependency.
"""
from contextlib import contextmanager
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import sqlite3
import zipfile

from filelock import FileLock
import numpy as np

from surface_computation.em_identity import DIAGNOSTIC_DTYPES

FORMAT = 'demixing-raw-corpus/1'
DTYPES = {'bias':'float32', **{'em_'+k:v for k,v in DIAGNOSTIC_DTYPES.items()},
          'em_parameters':'float32'}


def signature_digest(signature):
    return hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temporary,path)


def array_shape(name, rows, simulations):
    tail = (2,) if name=='bias' else (5,2) if name=='em_parameters' else ()
    return (rows,simulations)+tail


def payload_bytes(rows,simulations):
    return int(sum(np.prod(array_shape(name,rows,simulations))*np.dtype(dtype).itemsize
                   for name,dtype in DTYPES.items()))


def tile_count(signature):
    return ((signature['n_rows']+signature['shard_rows']-1)//signature['shard_rows']
            *((signature['n_simulations']+signature['simulation_chunk']-1)//signature['simulation_chunk']))


def require_space(directory,required):
    if shutil.disk_usage(directory).free < required+16*1024**2:
        raise OSError(f'output needs {required/1e9:.3f} GB plus checkpoint metadata')


def release_pages(array):
    """Keep mapped-page RSS bounded; the filesystem cache remains reusable."""
    mapping = getattr(array,'_mmap',None)
    if mapping is not None and hasattr(mapping,'madvise'):
        mapping.madvise(mmap.MADV_DONTNEED)


def validate_tile(arrays, rows, simulations):
    for name,dtype in DTYPES.items():
        value = arrays[name]
        if value.shape!=array_shape(name,rows,simulations) or value.dtype!=np.dtype(dtype):
            raise ValueError(f'invalid corpus tile array {name}: {value.shape}, {value.dtype}')
        if not np.isfinite(value).all():
            raise ValueError(f'nonfinite corpus tile array {name}')
    if not arrays['em_converged'].all() or arrays['em_nonfinite_starts'].any() or arrays['em_material_decreases'].any():
        raise ValueError('corpus tile failed EM validation')


def tile_digest(signature, coordinates, arrays):
    digest = hashlib.sha256((signature+str(tuple(coordinates))).encode())
    for name in DTYPES:
        digest.update(np.ascontiguousarray(arrays[name]).view(np.uint8))
    return digest.hexdigest()


def summary(arrays):
    names = dict(selected_cap_hits=~arrays['em_converged'],
        capped_losing_starts=arrays['em_capped_starts'],
        nonfinite_starts=arrays['em_nonfinite_starts'],
        material_decreases=arrays['em_material_decreases'],
        continued_winners=arrays['em_continued_winner'])
    return dict({k:int(np.sum(v,dtype=np.int64)) for k,v in names.items()},
                maximum_iterations=int(arrays['em_iterations'].max()))


class Corpus:
    def __init__(self, root, *, writable=False):
        self.root = Path(root)
        self.signature = json.loads((self.root/'manifest.json').read_text())
        self.identity = signature_digest(self.signature)
        if self.signature.get('storage_format')!=FORMAT:
            raise ValueError('unsupported simulation corpus format')
        self.rows = self.signature['n_rows']
        self.simulations = self.signature['n_simulations']
        self.design = np.load(self.root/'design.npy',mmap_mode='r')
        self.strata = json.loads((self.root/'strata.json').read_text())
        if self.design.shape!=(self.rows,4) or (self.strata is not None and len(self.strata)!=self.rows):
            raise ValueError('invalid corpus design or strata')
        if hashlib.sha256(np.ascontiguousarray(self.design).view(np.uint8)).hexdigest()!=self.signature['design_sha256']:
            raise ValueError('corpus design differs from manifest')
        if hashlib.sha256(json.dumps(self.strata).encode()).hexdigest()!=self.signature['strata_sha256']:
            raise ValueError('corpus strata differ from manifest')
        self.arrays = {}
        for name,dtype in DTYPES.items():
            value = np.load(self.root/(name+'.npy'),mmap_mode='r+' if writable else 'r')
            if value.shape!=array_shape(name,self.rows,self.simulations) or value.dtype!=np.dtype(dtype):
                raise ValueError(f'invalid corpus array {name}')
            self.arrays[name] = value
        uri = (self.root/'tiles.sqlite3').resolve().as_uri()+'?mode='+('rw' if writable else 'ro')
        self.db = sqlite3.connect(uri,uri=True)
        if self.db.execute('SELECT signature FROM identity').fetchone()!=(self.identity,):
            raise ValueError('corpus tile index belongs to another run')
        self.meta = None
        if not writable:
            if not (self.root/'complete.json').exists():
                raise ValueError(f'simulation corpus is incomplete: {self.root}')
            marker = json.loads((self.root/'complete.json').read_text())
            if marker['signature']!=self.identity or marker['index_sha256']!=self.index_digest():
                raise ValueError('corpus is incomplete or its tile index changed')
            self.meta = marker['meta']

    def close(self):
        self.db.close()
        self.arrays.clear()

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()

    def coordinates(self, row_start, sim_start):
        return (row_start,min(row_start+self.signature['shard_rows'],self.rows),
                sim_start,min(sim_start+self.signature['simulation_chunk'],self.simulations))

    def tile(self, coordinates):
        a,b,c,d = coordinates
        return {name:value[a:b,c:d] for name,value in self.arrays.items()}

    def valid_tile(self, coordinates):
        a,b,c,d = coordinates
        saved = self.db.execute('SELECT row_stop,sim_stop,digest,summary FROM tiles WHERE row_start=? AND sim_start=?', (a,c)).fetchone()
        if saved is None:
            return False
        try:
            arrays = self.tile(coordinates)
            return (saved[:2]==(b,d) and saved[2]==tile_digest(self.identity,coordinates,arrays)
                    and json.loads(saved[3])==summary(arrays))
        finally:
            for array in self.arrays.values():
                release_pages(array)

    def write_tile(self, coordinates, arrays):
        a,b,c,d = coordinates
        if coordinates!=self.coordinates(a,c) or a%self.signature['shard_rows'] or c%self.signature['simulation_chunk']:
            raise ValueError('invalid logical tile coordinates')
        validate_tile(arrays,b-a,d-c)
        expected = tile_digest(self.identity,coordinates,arrays)
        (self.root/'complete.json').unlink(missing_ok=True)
        with self.db:
            self.db.execute('DELETE FROM tiles WHERE row_start=? AND sim_start=?',(a,c))
        for name,value in self.arrays.items():
            value[a:b,c:d] = arrays[name]
            value.flush()
            release_pages(value)
        if tile_digest(self.identity,coordinates,self.tile(coordinates))!=expected:
            raise OSError('corpus tile write failed checksum verification')
        with self.db:
            self.db.execute('INSERT INTO tiles VALUES (?,?,?,?,?,?)',
                (a,b,c,d,expected,json.dumps(summary(arrays))))
        for value in self.arrays.values():
            release_pages(value)

    def index_digest(self):
        digest = hashlib.sha256()
        count = 0
        for a,b,c,d,checksum,encoded in self.db.execute('SELECT row_start,row_stop,sim_start,sim_stop,digest,summary FROM tiles ORDER BY row_start,sim_start'):
            if (a,b,c,d)!=self.coordinates(a,c) or not 0<=a<b<=self.rows or not 0<=c<d<=self.simulations or a%self.signature['shard_rows'] or c%self.signature['simulation_chunk']:
                raise ValueError('invalid corpus coverage')
            digest.update(json.dumps([a,b,c,d,checksum,encoded]).encode())
            count += 1
        expected = tile_count(self.signature)
        if count!=expected:
            raise ValueError(f'incomplete corpus coverage: {count}/{expected} tiles')
        return digest.hexdigest()

    def verify(self):
        """Explicit full-payload integrity scan, bounded to one logical tile."""
        self.index_digest()
        for coordinates in self.db.execute('SELECT row_start,row_stop,sim_start,sim_stop FROM tiles'):
            if not self.valid_tile(coordinates):
                raise ValueError(f'corpus tile checksum failed: {coordinates}')

    def finish(self, meta):
        index = self.index_digest()
        total = {}
        for (encoded,) in self.db.execute('SELECT summary FROM tiles'):
            for name,value in json.loads(encoded).items():
                total[name] = max(total.get(name,0),value) if name=='maximum_iterations' else total.get(name,0)+value
        meta = dict(meta,generation_signature=self.signature,em_summary=total,n_nonfinite=0)
        atomic_json(self.root/'complete.json',dict(format=FORMAT,signature=self.identity,index_sha256=index,meta=meta))
        self.meta = meta


@contextmanager
def writer(root, signature, design, strata):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    with FileLock(root/'.writer.lock',timeout=0):
        manifest = root/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!=signature:
                raise ValueError(f'existing shard manifest does not match this run: {manifest}')
        else:
            if any(p.name!='.writer.lock' for p in root.iterdir()):
                raise ValueError(f'existing shards have no manifest: {root}')
            require_space(root,payload_bytes(len(design),signature['n_simulations'])+1024*tile_count(signature))
            # Publish identity last: an interrupted setup cannot be mistaken for
            # a resumable run with trustworthy arrays and an index.
            np.save(root/'design.npy',design)
            atomic_json(root/'strata.json',strata)
            for name,dtype in DTYPES.items():
                value = np.lib.format.open_memmap(root/(name+'.npy'),mode='w+',dtype=dtype,
                    shape=array_shape(name,len(design),signature['n_simulations']))
                del value
            with sqlite3.connect(root/'tiles.sqlite3') as db:
                db.execute('CREATE TABLE identity (signature TEXT NOT NULL)')
                db.execute('INSERT INTO identity VALUES (?)',(signature_digest(signature),))
                db.execute('CREATE TABLE tiles (row_start INTEGER,row_stop INTEGER,sim_start INTEGER,sim_stop INTEGER,digest TEXT,summary TEXT,PRIMARY KEY(row_start,sim_start))')
            atomic_json(manifest,signature)
        with Corpus(root,writable=True) as corpus:
            yield corpus


def export_npz(root, destination, *, compress=False):
    """NumPy streams mapped arrays through bounded buffers into the ZIP64 NPZ."""
    destination = Path(destination)
    if destination.suffix!='.npz':
        raise ValueError('corpus export destination must end in .npz')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(Path(root)/'.writer.lock',timeout=0), Corpus(root) as corpus:
        temporary = destination.with_name('.'+destination.name+'.tmp')
        temporary.unlink(missing_ok=True)
        required = sum(value.nbytes for value in corpus.arrays.values())+corpus.design.nbytes
        require_space(destination.parent,required)
        arrays = dict(design=corpus.design,strata=np.asarray(corpus.strata or [],dtype=str),
                      meta=np.asarray(json.dumps(corpus.meta)),**corpus.arrays)
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED) as archive:
            for name,value in arrays.items():
                with archive.open(name+'.npy','w',force_zip64=True) as member:
                    np.lib.format.write_array_header_2_0(member,dict(
                        descr=np.lib.format.dtype_to_descr(value.dtype),fortran_order=False,shape=value.shape))
                    flat = value.reshape(-1)
                    count = max(1,16*1024**2//value.dtype.itemsize)
                    for start in range(0,flat.size,count):
                        member.write(flat[start:start+count].tobytes())
                        release_pages(value)
        os.replace(temporary,destination)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export a completed raw simulation corpus to NPZ')
    parser.add_argument('corpus',type=Path)
    parser.add_argument('out',type=Path,nargs='?')
    parser.add_argument('--compress',action='store_true')
    parser.add_argument('--verify',action='store_true',help='scan every tile checksum before export, or without an output')
    args = parser.parse_args()
    if args.out is None and not args.verify:
        parser.error('provide an NPZ output or --verify')
    if args.verify:
        with Corpus(args.corpus) as corpus:
            corpus.verify()
    if args.out is not None:
        export_npz(args.corpus,args.out,compress=args.compress)

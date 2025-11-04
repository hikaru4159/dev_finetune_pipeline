
import os
import capnp
import bz2
import zstandard as zstd

def load_log_capnp():
    capnp_path = os.path.join(os.path.dirname(__file__), '../../../cereal/log.capnp')
    return capnp.load(capnp_path)

class LogReader:
    def __init__(self, rlog_path):
        self._ents = []
        ext = os.path.splitext(rlog_path)[1]
        with open(rlog_path, 'rb') as f:
            dat = f.read()
        # bz2解凍
        if ext == ".bz2" or (dat and dat.startswith(b'BZh9')):
            dat = bz2.decompress(dat)
        # zst解凍
        elif ext == ".zst":
            dctx = zstd.ZstdDecompressor()
            try:
                dat = dctx.decompress(dat)
            except zstd.ZstdError:
                with dctx.stream_reader(dat) as reader:
                    dat = reader.read()
        self._capnp_log = load_log_capnp()
        ents = self._capnp_log.Event.read_multiple_bytes(dat)
        for e in ents:
            self._ents.append(e)

    def __iter__(self):
        return iter(self._ents)


# from torch.utils.ffi import _wrap_function
from ..ffi_ext import _wrap_function        # from torch 0.4.1 for debug in higher torch version
                                            # see https://codeantenna.com/a/7XO9h4MfdO
from ._resample2d import lib as _lib, ffi as _ffi

__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        if callable(fn):
            locals[symbol] = _wrap_function(fn, _ffi)
        else:
            locals[symbol] = fn
        __all__.append(symbol)

_import_symbols(locals())

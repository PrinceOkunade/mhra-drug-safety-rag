"""Fixed-size chunking windows must respect size and overlap bounds."""
from src.chunk.fixed import _window


def test_window_sizes_and_overlap():
    ids = list(range(1000))
    size, overlap = 512, 50
    windows = list(_window(ids, size, overlap))

    assert all(len(w) <= size for w in windows)      # never exceed size
    assert len(windows[0]) == size                   # first window is full
    assert windows[1][0] == size - overlap           # stride = size - overlap
    assert windows[-1][-1] == ids[-1]                # coverage reaches the end


def test_window_short_input_is_single_chunk():
    ids = list(range(100))
    windows = list(_window(ids, 512, 50))
    assert len(windows) == 1
    assert windows[0] == ids

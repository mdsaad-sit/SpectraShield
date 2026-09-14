import numpy as np
from app.inference.chunking import chunk_audio, get_chunk_count, get_chunk_count_live


def test_recorded_20sec_produces_4_chunks():
    """Recorded: 20 seconds at 16kHz with 5-second chunks = 4 chunks."""
    count = get_chunk_count(20, sr=16000)
    assert count == 4, f"Expected 4 recorded chunks for 20s, got {count}"


def test_recorded_23sec_produces_5_chunks_with_padding():
    """Recorded: 23 seconds should produce 5 chunks, final partial padded."""
    sr = 16000
    duration = 23.0
    num_samples = int(duration * sr)
    y = np.random.randn(num_samples).astype(np.float32)
    
    chunks = chunk_audio(y, sr=sr, chunk_duration=5)
    assert len(chunks) == 5, f"Expected 5 recorded chunks for 23s, got {len(chunks)}"
    
    # Last chunk should be padded to exactly 80000 samples
    _, start, end, chunk_y = chunks[4]
    assert len(chunk_y) == 80000, f"Final padded chunk should have 80000 samples, got {len(chunk_y)}"


def test_recorded_5sec_produces_1_chunk():
    """Recorded: exactly 5 seconds should produce 1 chunk."""
    count = get_chunk_count(5, sr=16000)
    assert count == 1, f"Expected 1 recorded chunk for 5s, got {count}"


def test_recorded_10sec_produces_2_chunks():
    """Recorded: 10 seconds should produce 2 chunks."""
    count = get_chunk_count(10, sr=16000)
    assert count == 2, f"Expected 2 recorded chunks for 10s, got {count}"


def test_live_20sec_produces_5_chunks():
    """Live: 20 seconds at 16kHz with 4-second chunks = 5 chunks."""
    count = get_chunk_count_live(20)
    assert count == 5, f"Expected 5 live chunks for 20s, got {count}"


def test_live_8sec_produces_2_chunks():
    """Live: 8 seconds should produce 2 chunks."""
    count = get_chunk_count_live(8)
    assert count == 2, f"Expected 2 live chunks for 8s, got {count}"


def test_live_4sec_produces_1_chunk():
    """Live: exactly 4 seconds should produce 1 chunk."""
    count = get_chunk_count_live(4)
    assert count == 1, f"Expected 1 live chunk for 4s, got {count}"


def test_live_21sec_produces_6_chunks():
    """Live: 21 seconds should produce 6 chunks (5 full + 1 partial padded)."""
    count = get_chunk_count_live(21)
    assert count == 6, f"Expected 6 live chunks for 21s, got {count}"


def test_live_24sec_produces_6_chunks():
    """Live: 24 seconds should produce exactly 6 chunks (no padding needed)."""
    count = get_chunk_count_live(24)
    assert count == 6, f"Expected 6 live chunks for 24s, got {count}"


def test_live_16sec_no_result_window():
    """Live: 16 seconds (4 chunks) should NOT have a complete 5-chunk window."""
    count = get_chunk_count_live(16)
    assert count == 4, f"Expected 4 live chunks for 16s, got {count}"
    # With only 4 chunks, no final result should be displayed yet


def test_live_20sec_first_result_window():
    """Live: 20 seconds (5 chunks) should have the first complete window."""
    count = get_chunk_count_live(20)
    assert count == 5, f"Expected 5 live chunks for 20s to get first result, got {count}"
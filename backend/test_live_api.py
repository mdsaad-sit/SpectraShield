import requests
import numpy as np
import uuid

BASE_URL = "http://127.0.0.1:5000"
SESSION_ID = f"test-{uuid.uuid4().hex[:8]}"

SAMPLE_RATE = 16000
CHUNK_DURATION = 4
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION

print("=" * 60)
print("SpectraShield Live API Test")
print("=" * 60)

# --------------------------------------------------
# 1. START SESSION
# --------------------------------------------------

response = requests.post(
    f"{BASE_URL}/api/start",
    json={"session_id": SESSION_ID}
)

print("\nSTART SESSION")
print(response.status_code)
print(response.json())

if response.status_code != 200:
    raise SystemExit("Failed to start session")

# --------------------------------------------------
# 2. SEND 7 CHUNKS
# --------------------------------------------------

for chunk_index in range(7):

    # Generate a 4-second test audio chunk
    audio = np.random.normal(
        0,
        0.05,
        CHUNK_SAMPLES
    ).astype(np.float32)

    payload = {
        "session_id": SESSION_ID,
        "audio_data": audio.tolist(),
        "sample_rate": SAMPLE_RATE,
        "chunk_index": chunk_index
    }

    response = requests.post(
        f"{BASE_URL}/api/predict_chunk",
        json=payload
    )

    print(f"\n{'=' * 60}")
    print(f"CHUNK {chunk_index + 1}")
    print("=" * 60)

    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        print("ERROR:")
        print(response.text)
        break

    result = response.json()

    print("Elapsed time:", result.get("elapsed_time"))
    print("Chunk prediction:", result.get("prediction"))
    print("Fake probability:", result.get("fake_probability"))
    print("Total accumulated:", result.get("total_chunks"))
    print("Final available:", result.get("final_result_available"))
    print("Final prediction:", result.get("final_prediction"))

    if result.get("final_result_available"):
        print("Real chunks:", result.get("real_chunks"))
        print("Fake chunks:", result.get("fake_chunks"))
        print("Confidence:", result.get("confidence"))
        print("Voting scope:", result.get("voting_scope"))

# --------------------------------------------------
# 3. STOP SESSION
# --------------------------------------------------

response = requests.post(
    f"{BASE_URL}/api/stop",
    json={"session_id": SESSION_ID}
)

print(f"\n{'=' * 60}")
print("STOP SESSION")
print("=" * 60)

print(response.status_code)
print(response.json())

print("\nTEST COMPLETE")
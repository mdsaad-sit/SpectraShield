import logging
import os

logger = logging.getLogger(__name__)


def validate_audio_file(file):
    """Validate uploaded audio file.

    Args:
        file: Flask FileStorage object

    Returns:
        (is_valid, error_message)
    """
    if not file:
        return False, "No file provided"

    filename = file.filename
    if not filename:
        return False, "No filename"

    # Check extension
    allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        return False, f"Unsupported file extension: {ext}"

    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning

    max_size = 100 * 1024 * 1024  # 100MB
    if file_size > max_size:
        return False, f"File too large: {file_size / (1024*1024):.1f}MB (max: 100MB)"

    # Check empty file
    if file_size == 0:
        return False, "Empty file"

    return True, None
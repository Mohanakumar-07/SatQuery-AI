import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from ml.data.build_bigearthnet_manifest import download_url_to_file_with_resume


class FakeResponse:
    def __init__(self, chunks, fail_after=None, status_code=200, headers=None):
        self.chunks = chunks
        self.fail_after = fail_after
        self.status_code = status_code
        self.headers = headers or {}
        self._iter_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, chunk_size=1 << 20):
        for chunk in self.chunks:
            self._iter_count += 1
            if self.fail_after is not None and self._iter_count >= self.fail_after:
                raise requests.exceptions.ChunkedEncodingError("incomplete stream")
            yield chunk

    def raise_for_status(self):
        return None


class DownloadStreamRetriesTest(unittest.TestCase):
    def test_retries_on_chunked_encoding_error_with_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "archive.bin"
            responses = [
                FakeResponse([b"ab", b"c", b"def"], fail_after=3, status_code=206, headers={"Range": "bytes=0-"}),
                FakeResponse([b"def", b"ghi"], status_code=206, headers={"Range": "bytes=3-"}),
            ]

            def fake_get(url, stream=True, timeout=120, headers=None):
                self.assertIsInstance(headers, dict)
                return responses.pop(0)

            with patch("requests.get", side_effect=fake_get):
                downloaded = download_url_to_file_with_resume("https://example.com/archive", dest, timeout=30, max_retries=2)

            self.assertEqual(downloaded.read_bytes(), b"abcdefghi")


if __name__ == "__main__":
    unittest.main()

from run_download_llm import llm_downloader
import unittest
from config import TEST_DATA_DIR


class TestDownloadLLM(unittest.TestCase):

    def setUp(self):
        import argparse
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory(prefix='test_download_llm_')
        self.args = argparse.Namespace(
            repo_id="mse30/bart-base-finetuned-pubmed",
            repo_type=None,
            revision="2dcf6798d3889087bf314087dfc84a22c92e26d1",
            local_dir=self.temp_dir.name,
            local_dir_use_symlinks=None,
            authentication_token=None,
            allow_patterns='*.json'
        )

    def test_llm_downloader(self):
        llm_downloader(self.args)
        self.assertRaises(FileExistsError, llm_downloader, self.args)

    def test_checksum(self):
        import os
        import hashlib
        import json
        with open(os.path.join(TEST_DATA_DIR, "md5_checks.json"), "r") as f:
            file_md5_digests = json.load(f)

        for file in os.listdir(self.temp_dir.name):
            file_path = os.path.join(self.temp_dir.name, file)
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            self.assertEqual(file_hash, file_md5_digests[file])

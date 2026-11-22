"""Exercise the repository hooks through Git, including partially staged files."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MAIL = "123+test-user@users.noreply.github.com"


class CommitPrivacyTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)
        self.env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
        for key in tuple(self.env):
            if key.startswith(("GIT_AUTHOR_", "GIT_COMMITTER_")):
                del self.env[key]
        self.git("init", "-q")
        self.git("config", "user.name", "test-user")
        self.git("config", "user.email", MAIL)
        self.git("config", "core.hooksPath", ".githooks")
        shutil.copytree(ROOT / ".githooks", self.repo / ".githooks")
        (self.repo / "scripts").mkdir()
        shutil.copy(ROOT / "scripts/check_commit_privacy.py", self.repo / "scripts")
        shutil.copy(ROOT / "CONTRIBUTING.md", self.repo)
        shutil.copy(ROOT / ".gitignore", self.repo)
        (self.repo / "tests").mkdir()
        shutil.copy(__file__, self.repo / "tests")
        self.git("add", ".")
        self.git("commit", "-qm", "Enable local checks")

    def git(self, *args, ok=True, env=None):
        result = subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True,
                                env={**self.env, **(env or {})})
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        return result

    def stage(self, path, value):
        (self.repo / path).write_text(value, encoding="utf-8")
        self.git("add", "--", path)

    def test_effective_identities_and_message(self):
        private = "fixture-person" + "@" + "example.org"
        self.stage("work.txt", "A public result.\n")
        for args, env in [
            (["--author", "fixture-person <" + private + ">"], {}),
            ([], {"GIT_COMMITTER_EMAIL": private}),
        ]:
            err = self.git("commit", "-qm", "Private identity", *args, ok=False, env=env).stderr
            self.assertIn("noreply", err)
            self.assertNotIn(private, err)
        err = self.git("commit", "-qm", "Research\n\nCo-authored-by: A <" + private + ">",
                       ok=False).stderr
        self.assertIn("attribution trailer", err)
        self.assertNotIn(private, err)
        self.git("commit", "-qm", "Research\n\nCo-authored-by: A <" + MAIL + ">")

    def test_staged_snapshot_and_private_filename(self):
        token = "ghp_" + "A" * 36
        self.stage("work.txt", "safe research result\n")
        (self.repo / "work.txt").write_text(token)
        self.git("commit", "-qm", "Use staged result")
        self.git("add", "work.txt")
        (self.repo / "work.txt").write_text("safe")
        err = self.git("commit", "-qm", "Capture results", ok=False).stderr
        self.assertIn("provider credential", err)
        self.assertNotIn(token, err)
        self.git("reset", "-q", "HEAD", "work.txt")
        name = "test-" + token + ".log"
        self.stage(name, "test")
        err = self.git("commit", "-qm", "Capture output", ok=False).stderr
        self.assertIn("<private filename>", err)
        self.assertNotIn(token, err)

    def test_notebook_privacy_without_rewriting_legacy_outputs(self):
        local_home = "/ho" + "me/fixture-person"
        notebook = {
            "cells": [{"source": ["print(1)"], "outputs": [{"text": [local_home + "/job"]}]}],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
        }
        self.stage("example.ipynb", json.dumps(notebook))
        self.git("-c", "core.hooksPath=.git/no-hooks", "commit", "-qm", "Seed legacy artifact")
        notebook["cells"][0]["source"] = ["# Public provenance: https://example.edu/paper\n", "print(2)"]
        self.stage("example.ipynb", json.dumps(notebook, indent=2))
        self.git("commit", "-qm", "Update calculation")
        notebook["metadata"]["workspace"] = local_home + "/new-job"
        self.stage("example.ipynb", json.dumps(notebook))
        err = self.git("commit", "-qm", "Add metadata", ok=False).stderr
        self.assertIn("workstation home path", err)
        self.assertNotIn(local_home, err)

    def test_identifiers_and_credentials_are_private(self):
        term = "fixture" + "-private-team"
        (self.repo / ".git/info/privacy-identifiers").write_text(term + "\n")
        self.stage("notes.md", "https://example.org/" + term + "/project\n")
        err = self.git("commit", "-qm", "Write notes", ok=False).stderr
        self.assertIn("locally marked private identifier", err)
        self.assertNotIn(term, err)
        self.stage("notes.md", "Public citation and institutional contact research@example.edu.\n")
        err = self.git("commit", "-qm", "Update " + term, ok=False).stderr
        self.assertIn("commit message", err)
        self.assertNotIn(term, err)
        err = self.git("commit", "-qm", "Update notes", "--author", term + " <" + MAIL + ">",
                       ok=False).stderr
        self.assertIn("author identity", err)
        self.assertNotIn(term, err)
        self.git("commit", "-qm", "Document public sources")

    def stage_office(self, path, creator, editor, body="Public research result", part="docProps/core.xml"):
        props = ET.Element("{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}coreProperties")
        ET.SubElement(props, "{http://purl.org/dc/elements/1.1/}creator").text = creator
        ET.SubElement(props, "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastModifiedBy").text = editor
        rels = ET.Element("Relationships")
        ET.SubElement(rels, "Relationship", Id="r1", Target=part,
                      Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties")
        with zipfile.ZipFile(self.repo / path, "w", zipfile.ZIP_DEFLATED) as package:
            package.writestr(part, ET.tostring(props, encoding="utf-8"))
            package.writestr("_rels/.rels", ET.tostring(rels, encoding="utf-8"))
            package.writestr("ppt/slides/slide1.xml", body)
        self.git("add", "--", path)

    def test_compressed_office_authors_and_public_provenance(self):
        synthetic_name = "Fictional Tester"
        private_email = "fixture-mailbox" + "@" + "example.org"
        private_term = "fixture-private-team"
        (self.repo / ".git/info/privacy-identifiers").write_text(private_term + "\n")
        for name, creator, editor, expected in [
            ("slides.pptx", synthetic_name, "", "personal author name"),
            ("document.docx", "Apache POI", private_email, "author email"),
            ("sheet.xlsx", "Research Team", private_term, "locally marked private identifier"),
        ]:
            with self.subTest(name=name):
                self.stage_office(name, creator, editor, part="props/publication.xml")
                err = self.git("commit", "-qm", "Update artifact", ok=False).stderr
                self.assertIn(expected, err)
                for value in [synthetic_name, private_email, private_term]:
                    self.assertNotIn(value, err)
                self.git("restore", "--staged", "--", name)
        self.stage_office("slides.pptx", "Research Team", "GitHub Actions",
                          body="Public citation: " + synthetic_name)
        self.git("commit", "-qm", "Publish cited result")

    def test_office_edits_preserve_legacy_metadata(self):
        self.stage_office("slides.pptx", "Fictional Tester", "LibreOffice")
        self.git("-c", "core.hooksPath=.git/no-hooks", "commit", "-qm", "Seed old presentation")
        self.stage_office("slides.pptx", "Fictional Tester", "LibreOffice", body="Updated results")
        self.git("commit", "-qm", "Correct slide content")
        self.stage_office("slides.pptx", "Fictional Tester", "Second Example", body="Updated results")
        err = self.git("commit", "-qm", "Save presentation", ok=False).stderr
        self.assertIn("lastModifiedBy", err)
        self.assertNotIn("Second Example", err)
        self.stage_office("slides.pptx", "OpenKernel Team", "", body="Updated results")
        self.git("commit", "-qm", "Clear old authors")

    def test_oversized_compressed_office_properties_require_inspection(self):
        self.stage_office("slides.pptx", "x" * (256 * 1024 + 1), "Research Team")
        err = self.git("commit", "-qm", "Include metadata", ok=False).stderr
        self.assertIn("Office author metadata could not be verified", err)

    def test_forced_add_of_ignored_credentials(self):
        shutil.copy(ROOT / ".gitignore", self.repo / ".gitignore")
        (self.repo / "kaggle.json").write_text("{}")
        self.git("add", "-f", "kaggle.json")
        err = self.git("commit", "-qm", "Include service config", ok=False).stderr
        self.assertIn("credential file", err)


if __name__ == "__main__":
    unittest.main()

"""Check Git commit identities, messages and new staged disclosures, without logging values."""

import collections
import io
import json
import os
from pathlib import Path
import posixpath
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile


NOREPLY = re.compile(r"(?:[^@\s<>]+@users\.noreply\.github\.com|noreply@github\.com)", re.I)
PATTERNS = {
    "provider credential": re.compile(
        r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{40,255}|"
        r"sk-(?:proj-|org-)?[A-Za-z0-9_-]{24,255}|hf_[A-Za-z0-9]{24,128}|"
        r"KGAT_[A-Za-z0-9_-]{20,255}|AIza[A-Za-z0-9_-]{35}|AKIA[A-Z0-9]{16})(?![A-Za-z0-9])"
    ),
    "private signing key": re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    "private mailbox": re.compile(
        r"[A-Z0-9._%+-]+@(?:gmail\.com|googlemail\.com|hotmail\.[A-Z.]+|outlook\.[A-Z.]+|"
        r"live\.com|icloud\.com|me\.com|yahoo\.[A-Z.]+|proton(?:mail)?\.(?:com|me)|"
        r"yandex\.(?:ru|com)|mail\.ru)\b", re.I
    ),
    "credential in URL": re.compile(r"\b(?:https?|postgres(?:ql)?|mysql)://[^\s/:@]+:[^\s/@]+@", re.I),
    "literal service key": re.compile(
        r"""\b(?:KAGGLE_KEY|KAGGLE_API_TOKEN|OPENAI_API_KEY|ANTHROPIC_API_KEY|GITHUB_TOKEN|GH_TOKEN|HF_TOKEN|AWS_SECRET_ACCESS_KEY)\b["']?\s*[:=]\s*["']?([a-zA-Z0-9+/_=-]{16,255})(?![\w])"""
    ),
}
HOME_PATH = re.compile(
    r"(?<![\w/])(?:/(?:home|Users)/|[A-Z]:[\\/]+(?:Users|Documents and Settings)[\\/]+)"
    r"""([\w.-]+)(?:[\\/]+[^\s"'<>`\\/\x00-\x1f]+)*""", re.I
)
GENERIC_HOMES = {"runner", "user", "username", "ubuntu", "jovyan", "codespace", "vscode",
                 "kaggle", "public", "default", "container", "app", "root"}
CREDENTIAL_NAMES = {".netrc", "_netrc", ".pypirc", "kaggle.json", "id_rsa", "id_ed25519",
                    "id_dsa", "id_ecdsa"}
TRAILER = re.compile(r"^(?:co-authored-by|signed-off-by):.*?<([^<>\s]+)>", re.I | re.M)
STRUCTURED_SUFFIXES = {".ipynb", ".json"}
OFFICE_SUFFIXES = {".docx", ".pptx", ".xlsx"}
MAX_OFFICE_BYTES = 100 * 1024 * 1024
MAX_DIRECTORY_BYTES = 4 * 1024 * 1024
MAX_PROPERTY_BYTES = 256 * 1024
OFFICE_AUTHORS = {"creator", "lastModifiedBy"}
EMAIL = re.compile(r"[A-Z0-9._%+\[\]-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
GENERIC_AUTHOR_WORDS = {
    "actions", "anonymous", "apache", "apple", "aspose", "assistant", "automation", "bot",
    "builder", "challenge", "collaboration", "contributor", "contributors", "document", "editor",
    "foundation", "generator", "github", "google", "group", "jupyter", "kaggle", "kernel", "lab", "labs",
    "libreoffice", "microsoft", "notebook", "office", "openai", "openkernel", "powerpoint",
    "openoffice", "openpyxl", "openxml", "pandoc", "project", "research", "service", "slides",
    "software", "staff", "system", "team", "tool", "user",
}



def git(*args):
    return subprocess.run(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True).stdout


def private_identifiers():
    path = Path(os.fsdecode(git("rev-parse", "--git-path", "info/privacy-identifiers").rstrip(b"\n")))
    if not path.exists():
        return []
    return [re.compile(r"(?<!\w)" + re.escape(s) + r"(?!\w)", re.I)
            for line in path.read_text(encoding="utf-8").splitlines()
            if (s := line.strip()) and not s.startswith("#")]


def matches(text, identifiers):
    hits = collections.Counter()
    for kind, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group(match.lastindex or 0)
            hits[kind, value] += 1
    for match in HOME_PATH.finditer(text):
        if match[1].casefold() not in GENERIC_HOMES:
            hits["workstation home path", match[0]] += 1
    for pattern in identifiers:
        for match in pattern.finditer(text):
            hits["locally marked private identifier", match[0].casefold()] += 1
    return hits


def json_text(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str):
                yield key + "=" + item
            else:
                yield key
                yield from json_text(item)
    elif isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            yield "".join(value)
        else:
            for item in value:
                yield from json_text(item)


def decode(raw):
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", "replace")
    return raw.decode("utf-8-sig", "replace")


class OfficeReader(io.BytesIO):
    """Bound individual ZIP reads, including the uncompressed central directory."""

    def read(self, size=-1):
        requested = len(self.getbuffer()) - self.tell() if size < 0 else size
        if requested > MAX_DIRECTORY_BYTES:
            raise ValueError("oversize ZIP read")
        return super().read(size)


def office_xml(archive, member):
    if member.file_size > MAX_PROPERTY_BYTES:
        raise ValueError("oversize Office property part")
    with archive.open(member) as source:
        data = source.read(MAX_PROPERTY_BYTES + 1)
    if len(data) > MAX_PROPERTY_BYTES or b"<!DOCTYPE" in data.replace(b"\x00", b"").upper():
        raise ValueError("unsupported Office property XML")
    return ET.fromstring(data)


def personal_author(value):
    words = value.replace(",", " ").split()
    return (2 <= len(words) <= 6
            and all(re.fullmatch(r"[^\W\d_][^\W\d_.'’-]*(?:[.'’-][^\W\d_]*)*", word)
                    for word in words)
            and not any(word.casefold() in GENERIC_AUTHOR_WORDS for word in words))


def office_hits(oid, identifiers):
    hits = collections.Counter()
    try:
        if int(git("cat-file", "-s", oid)) > MAX_OFFICE_BYTES:
            raise ValueError("oversize Office package")
        with zipfile.ZipFile(OfficeReader(git("cat-file", "blob", oid))) as archive:
            core = {"docProps/core.xml"}
            names = collections.Counter(archive.namelist())
            if "_rels/.rels" in names:
                if names["_rels/.rels"] != 1:
                    raise ValueError("ambiguous Office relationships")
                for rel in office_xml(archive, archive.getinfo("_rels/.rels")):
                    if not rel.get("Type", "").endswith("/metadata/core-properties"):
                        continue
                    target = urlsplit(rel.get("Target", ""))
                    if target.scheme or target.netloc or rel.get("TargetMode") == "External":
                        raise ValueError("external Office core properties")
                    part = posixpath.normpath(unquote(target.path).lstrip("/"))
                    if part not in names:
                        raise ValueError("missing Office core properties")
                    core.add(part)
                    if len(core) > 4:
                        raise ValueError("too many Office core property parts")
            for part in core.intersection(names):
                if names[part] != 1:
                    raise ValueError("duplicate Office core properties")
                for element in office_xml(archive, archive.getinfo(part)).iter():
                    field = element.tag.rsplit("}", 1)[-1]
                    if field not in OFFICE_AUTHORS:
                        continue
                    value = "".join(element.itertext()).strip()
                    for (kind, match), count in matches(value, identifiers).items():
                        hits[f"Office {field}: {kind}", match] += count
                    for email in EMAIL.findall(value):
                        if not NOREPLY.fullmatch(email):
                            hits[f"Office {field}: author email", email.casefold()] += 1
                    for name in value.split(";"):
                        if personal_author(name):
                            hits[f"Office {field}: personal author name", name.casefold().strip()] += 1
    except (zipfile.BadZipFile, ET.ParseError, OSError, RuntimeError, ValueError, NotImplementedError):
        # New or changed unreadable packages need inspection. The object ID
        # distinguishes them while keeping diagnostics free of matched values.
        hits["Office author metadata could not be verified", oid] += 1
    return hits


def blob_hits(oid, mode, filename, identifiers):
    if mode not in {b"100644", b"100755", b"120000"}:
        return collections.Counter()
    suffix = Path(filename).suffix.lower()
    if suffix in OFFICE_SUFFIXES and mode != b"120000":
        return office_hits(oid, identifiers)
    text = decode(git("cat-file", "blob", oid))
    if suffix in STRUCTURED_SUFFIXES:
        try:
            values = json_text(json.loads(text))
            hits = collections.Counter()
            for value in values:
                hits.update(matches(value, identifiers))
            return hits
        except (ValueError, RecursionError):
            pass  # A malformed JSON file still receives the raw text check.
    return matches(text, identifiers)


def credential_path(path):
    name = Path(path).name.lower()
    return (name in CREDENTIAL_NAMES or name == ".env" or
            (name.startswith(".env.") and name not in {".env.example", ".env.sample", ".env.template"})
            or name.endswith((".p12", ".pfx")) or path.lower().endswith("/.aws/credentials")
            or path.lower() == ".aws/credentials")


def staged_issues(identifiers):
    # Read blobs from the index. A partially staged file may differ from disk.
    # Compare each sensitive value's multiplicity with its own parent version.
    raw = git("diff", "--cached", "--raw", "--no-abbrev", "--find-renames", "--no-ext-diff",
              "--diff-filter=ACMRT", "-z", "--")
    fields = iter(raw.split(b"\0")[:-1])
    for record in fields:
        old_mode, new_mode, old_oid, new_oid, status = record[1:].split()
        old_path = os.fsdecode(next(fields))
        new_path = os.fsdecode(next(fields)) if status.startswith((b"R", b"C")) else old_path
        name_hits = matches(new_path, identifiers)
        safe_name = "<private filename>" if name_hits else ascii(new_path)
        if old_mode == b"000000" or new_path != old_path:
            if credential_path(new_path):
                yield f"{safe_name}: credential file"
            for kind in sorted({kind for kind, _ in name_hits}):
                yield f"<private filename>: {kind}"
        if new_oid == old_oid and Path(old_path).suffix == Path(new_path).suffix:
            continue
        old = blob_hits(old_oid, old_mode, old_path, identifiers)
        new = blob_hits(new_oid, new_mode, new_path, identifiers)
        for kind in sorted({kind for kind, _ in new - old}):
            yield f"{safe_name}: {kind}"


def identity_issues(identifiers):
    for role in ("AUTHOR", "COMMITTER"):
        ident = os.fsdecode(git("var", f"GIT_{role}_IDENT")).strip()
        address = re.search(r"<([^<>]+)> \d+ [+-]\d{4}$", ident)
        if not address or not NOREPLY.fullmatch(address[1]):
            yield f"{role.lower()} email must be a GitHub noreply address"
        if any(pattern.search(ident) for pattern in identifiers):
            yield f"{role.lower()} identity contains a locally marked private identifier"


def main(argv):
    identifiers = private_identifiers()
    if argv == ["staged"]:
        issues = [*identity_issues(identifiers), *staged_issues(identifiers)]
    elif len(argv) == 2 and argv[0] == "message":
        message = decode(Path(argv[1]).read_bytes())
        issues = [f"commit message: {kind}" for kind in sorted({k for k, _ in matches(message, identifiers)})]
        if any(not NOREPLY.fullmatch(email) for email in TRAILER.findall(message)):
            issues.append("commit attribution trailer must use a GitHub noreply address")
    else:
        raise ValueError("invalid check mode")
    for issue in issues:
        print(f"privacy check: {issue}", file=sys.stderr)
    return bool(issues)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (OSError, subprocess.CalledProcessError, ValueError, StopIteration):
        sys.exit("privacy check could not read the staged snapshot, Git identity, or message")

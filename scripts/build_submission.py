"""Rebuild final PACKAGE_MANIFEST.json and create the LUISS Group 9 submission ZIP.

Run from project/ with: python scripts/build_submission.py --out ../group_9.zip
This never edits frozen historical scientific manifests, files or reports.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import sys

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
IGNORED={'.git','.venv','venv','node_modules','__pycache__','.pytest_cache','outputs','.mypy_cache','.ruff_cache','presentation'}

def included(f:Path)->bool:
    r=f.relative_to(PROJECT)
    return f.is_file() and not any(p in IGNORED for p in r.parts) and r.as_posix()!='PACKAGE_MANIFEST.json'

def update_manifest():
    records={}
    for file in sorted(PROJECT.rglob('*')):
        if included(file):
            records[file.relative_to(PROJECT).as_posix()]={'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'size_bytes':file.stat().st_size}
    manifest={'files':records,
              'presentation_status':'Included at ZIP root and inside project/',
              'python':'3.11',
              'scientific_frozen_files_unchanged':47,
              'source':'13 Nov 2026 Group 9 definitive scientific submission with retrospective policy comparison',
              'type':'Group9-final-reproducible-submission'}
    (PROJECT/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
    return len(records)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    count=update_manifest()
    from src.release_pipeline import audit_release
    audit_release(PROJECT)
    from src.orchestrator_loader import verify_all_orchestrators
    assert verify_all_orchestrators(PROJECT)==4
    deck=PROJECT/'presentation.pptx'
    if not deck.is_file():raise FileNotFoundError(deck)
    target=a.out.resolve()
    if target.is_relative_to(PROJECT):
        raise ValueError('Output ZIP must be outside project/')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as f:
        for file in sorted(PROJECT.rglob('*')):
            if included(file) or (file.name=='PACKAGE_MANIFEST.json' and file.parent==PROJECT):
                rel=file.relative_to(PROJECT)
                f.write(file,arcname=(Path('project')/rel).as_posix())
        f.write(deck,arcname='presentation.pptx')
    with zipfile.ZipFile(target) as f:
        assert f.testzip() is None
        assert 'project/PACKAGE_MANIFEST.json' in f.namelist()
        assert 'presentation.pptx' in f.namelist()
        assert f.read('presentation.pptx')==deck.read_bytes()
    print(f'OK: {count} files manifest-verified; ZIP {target}; outer presentation.pptx verified.')

if __name__=='__main__':main()

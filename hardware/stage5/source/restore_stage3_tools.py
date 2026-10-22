"""Restore the recorded Ubuntu tool packages; optionally rebuild OpenSTA.

Prerequisites: Ubuntu 24.04 x86-64-compatible runtime, Python 3, g++, make,
dpkg-deb and zlib development headers. Packages are extracted locally.
The nextpnr 0.6 binary needs its chip database at /usr/share/nextpnr;
this script creates a symlink only when that path does not already exist.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import urllib.request

stage=Path(__file__).resolve().parents[1]
tool=stage.parent/'toolchain'
prefix=tool/'native'
usr=prefix/'usr'
p=argparse.ArgumentParser()
p.add_argument('--rebuild-opensta',action='store_true')
a=p.parse_args()
packages=json.loads((tool/'native_packages.json').read_text())
tcl=json.loads((tool/'tcl_package.json').read_text())
packages[tcl['Package']]=tcl
packages.update(json.loads((tool/'stage3_installed_packages.json').read_text()))
for name,package in packages.items():
    archive=tool/Path(package['Filename']).name
    if not archive.exists():
        urllib.request.urlretrieve('https://archive.ubuntu.com/ubuntu/'+package['Filename'],archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==package['SHA256'],name
    subprocess.run(['dpkg-deb','-x',str(archive),str(prefix)],check=True)
for path,target in [(usr/'share/verilator/bin/verilator_bin','../../../bin/verilator_bin'),
                    (usr/'bin/tclsh','tclsh8.6')]:
    if not path.exists():
        path.symlink_to(target)
chipdb=Path('/usr/share/nextpnr')
if not chipdb.exists():
    chipdb.symlink_to(usr/'share/nextpnr',target_is_directory=True)
else:
    assert (chipdb/'ecp5').is_dir(), 'Existing nextpnr chip database requires inspection.'
env=dict(os.environ)
env.update(PATH=str(usr/'bin')+':'+env['PATH'],
           LD_LIBRARY_PATH=str(usr/'lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH',''),
           TCL_LIBRARY=str(usr/'share/tcltk/tcl8.6'),
           SWIG_LIB=str(usr/'share/swig4.0'),
           BISON_PKGDATADIR=str(usr/'share/bison'), M4=str(usr/'bin/m4'))
if a.rebuild_opensta:
    provenance=json.loads((tool/'stage3_tool_provenance.json').read_text())
    for name in ['OpenSTA','cudd']:
        archive=tool/(name+'_source.tar.gz')
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==provenance[name]['archive_sha256']
        if not (tool/name).exists():
            with tarfile.open(archive) as f:
                f.extractall(tool,filter='data')
    for filename in ['aclocal.m4','configure','config.h.in','Makefile.in']:
        (tool/'cudd'/filename).touch()
    subprocess.run(['./configure','--enable-silent-rules','--disable-shared'],cwd=tool/'cudd',env=env,check=True)
    subprocess.run(['make','-j','3'],cwd=tool/'cudd',env=env,check=True)
    cmd=['cmake','-S',str(tool/'OpenSTA'),'-B',str(tool/'opensta_build'),
         '-DCMAKE_BUILD_TYPE=RelWithDebInfo','-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g0 -DNDEBUG',
         '-DCUDD_DIR='+str(tool/'cudd'),'-DCMAKE_PREFIX_PATH='+str(usr),
         '-DTCL_LIBRARY='+str(usr/'lib/x86_64-linux-gnu/libtcl8.6.so'),
         '-DTCL_INCLUDE_PATH='+str(usr/'include/tcl8.6'),'-DTCL_HEADER='+str(usr/'include/tcl8.6/tcl.h'),
         '-DEigen3_DIR='+str(usr/'share/eigen3/cmake'),'-DUSE_TCL_READLINE=OFF',
         '-DFLEX_INCLUDE_DIR='+str(usr/'include')]
    subprocess.run(cmd,env=env,check=True)
    subprocess.run(['cmake','--build',str(tool/'opensta_build'),'-j','3'],env=env,check=True)
print('Recorded tools are ready. Run the verification scripts from the stage3 directory.')

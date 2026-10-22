"""Inspect optional software/hardware dependencies without installing tools."""
from importlib import metadata,util
from pathlib import Path
import platform,shutil,json

def main():
    packages={}
    names={'numpy':'numpy','scipy':'scipy','pandas':'pandas','torch':'torch',
           'matplotlib':'matplotlib','PIL':'Pillow','sklearn':'scikit-learn','skimage':'scikit-image',
           'fontTools':'fonttools','pypdf':'pypdf','fitz':'PyMuPDF','reportlab':'reportlab'}
    for module,dist in names.items():
        found=util.find_spec(module) is not None
        try:version=metadata.version(dist) if found else None
        except metadata.PackageNotFoundError:version='installed; metadata unavailable'
        packages[module]={'available':found,'version':version}
    print(json.dumps({'python':platform.python_version(),'system':platform.platform(),
       'packages':packages,'executables':{x:shutil.which(x) for x in
       ['g++','git','iverilog','verilator','yosys','nextpnr-ecp5','sta']},
       'note':'Missing optional tools do not prevent source checks or GitHub upload. Full workflows require their own dependencies.'},indent=2))

if __name__=='__main__':main()

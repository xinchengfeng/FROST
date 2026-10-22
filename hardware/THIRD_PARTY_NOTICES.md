# Hardware third-party notices

Author-created FROST/SC adaptation RTL and experiment scripts retain their
existing status: no blanket author-code license has been selected by this
packaging task. Third-party licenses below remain in force independently.
Original files and notices have not been stripped or relicensed.

| Component | Locations | Origin / version | License |
|---|---|---|---|
| Berkeley HardFloat | `stage3/third_party/HardFloat-1/`, `stage5/third_party/HardFloat-1/` | Release1, 29July2019; http://www.jhauser.us/arithmetic/HardFloat.html | BSD 3-clause; retained `COPYING.txt` |
| Nangate Open Cell Library | `stage3/third_party/nangate45/`, `stage5/third_party/nangate45/` | `PDKv1.3_v2010_12.Apache.CCL`; retained README identifies OpenROAD adaptations and original SI2 source | Apache2.0; retained `LICENSE` |
| Nangate simulation-cell view | `*/third_party/nangate45/cells_sim.v` | Preserved evaluated generated/derivative cell model accompanying the above library | Library license retained with derivative; see original header and license |
| Original copied HardFloat modules | Under original source/RTL where applicable | Preserve upstream headers | Do not remove upstream BSD notices when modifying |

The Nangate library is a research/testing generic library and is explicitly
non-manufacturable. Only the small evaluated `.lib` and simulation-cell model
are bundled; a manufacturing PDK is not included. No proprietary vendor IP,
EDA binary, board package or licensed Arial font is redistributed. Source
versions, member paths and hashes are in `SOURCE_PROVENANCE.json`. Hashes for
the bundled third-party files can be independently checked there.

EDA tools are dependencies rather than bundled code. Consult their own
upstream licenses (Yosys, ABC, Verilator, nextpnr, OpenSTA, CUDD) when installing
or redistributing them. The original Ubuntu package and source metadata is
retained in `toolchain/` without distributing those binaries.

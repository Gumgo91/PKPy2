# Data provenance

The two CSV files in this directory are the public benchmark datasets used in
the PKPy2 manuscript's clinical analyses. They are mirrored from the PKGPT
repository at commit `ffe84301466297559a573a8641add1446c7dce8c`.

| Local file | Upstream file | SHA-256 |
|---|---|---|
| `data/theo.csv` | `dataset/theo.csv` | `67af92438a143b52206ba8c0f99179e0fd98c647b954dd14afdbf3c64bb946a9` |
| `data/warfarin.csv` | `dataset/wafarin.csv` | `5061a0724eed4d5f94a6d0284f6721101c9d941a365b942d8ef646c7830d52e5` |

The `warfarin.csv` local filename only corrects the spelling of the upstream
`wafarin.csv` path; no content transformation is applied. The theophylline
dataset is the classic `Theoph` pharmacokinetic dataset in NONMEM-style event
record format (`ID, TIME, AMT, DV, WT, SEX`); `AMT` is in mg/kg and is
multiplied by subject weight to obtain the administered dose in mg.

PKPy2's MIT license applies to the software in this repository. It does not
grant rights to third-party data; users are responsible for reviewing the
upstream terms and any original dataset conditions before redistribution.

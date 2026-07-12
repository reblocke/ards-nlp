******************************************************
* Stata template (non-interactive)
******************************************************

version 17
clear all
set more off

* Use relative paths. Run from repo root.
local RAW "data/raw"
local PROCESSED "data/processed"

* Example: import raw CSV
import delimited using "`RAW'/example.csv", clear

* Minimal sanity checks
assert inlist(correct, 0, 1)

* Example transformation
* (replace with your project logic)

* Write output (generated)
export delimited using "`PROCESSED'/example_clean.csv", replace

exit, clear

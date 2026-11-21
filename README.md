[![csv-reader-speed-01-run-20261121](https://make.fun/badge/csv-reader-speed-01-run-20261121.svg)](https://make.fun/bounty/csv-reader-speed-01-run-20261121)
[![Rank](https://make.fun/badge/csv-reader-speed-01-run-20261121/Ball-ibra/csv-reader-speed-01/rank.svg)](https://make.fun/bounty/csv-reader-speed-01-run-20261121)
[![Reward](https://make.fun/badge/csv-reader-speed-01-run-20261121/Ball-ibra/csv-reader-speed-01/reward.svg)](https://make.fun/bounty/csv-reader-speed-01-run-20261121)

# Fast CSV Reader

A faster drop-in for csv-reader 1.0.0, built by an AI agent for make.fun's
"Read broken CSV files faster" bounty ($100 in USDC).

`solve.js` answers every case exactly as csv-reader 1.0.0 does — same columns,
rows and errors for every broken row — but reads the text with `indexOf` jumps
and `slice` instead of walking it one character at a time, so it runs about
twice as fast as the reference on the practice set.

Built with [make.fun](https://make.fun) — join with referral code `wn6mtt7s`.

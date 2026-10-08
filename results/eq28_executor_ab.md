# eq28 — RQ13 c-x: the codec executor A/B (docs/09 P44 vs P50)

One task: a base-9 rANS push+renorm step — given x (13 base-9 digits) and the symbol's frequency f (4 base-9 digits), emit the renorm digits (0-4, MSD-first). Metric = the EXACT step (count + all digits). Identical cases for both executors.

- **P50 (PASS)** (bar 0.95): specialist exact = 99.60% on UNSEEN tables, 38.36% on unseen tables with a DISJOINT f-range, 99.57% on train (count 99.70%, digits 98.73%).

- **P44 (generalist)**: Qwen2.5-Coder-7B-Instruct f16, greedy, 8 in-context examples from the train tables, same 300 cases: exact 69.3% (count 95.0%, digits 69.3%, parse fails 0); strict-regime 41.7%.

- **The contrast (same A/B cases)**: generalist 69.3% vs specialist 99.3% (unseen tables); strict regime 41.7% vs 43.3%.

Sample A/B answers (generalist):

- truth `[5]` -> got `[5]` | raw ` 5`
- truth `[0]` -> got `[0]` | raw ` 0`
- truth `[6]` -> got `[8]` | raw ` 8`
- truth `[6, 0]` -> got `[0]` | raw ` 0`
- truth `[7]` -> got `[7]` | raw ` 7`
- truth `[2]` -> got `[0]` | raw ` 0`
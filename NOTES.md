# Notes

Durable context that doesn't belong in `TODO.md` (which tracks pending work).

## Project structure

This repo (`demixing_model`, DM) is one of three independent repos in this
project (see `/workspaces/demixing_model/AGENTS.md`): DM (this repo),
`bayesian_biases_zoo` (BBZ, the alternative-model zoo), and
`bias_model_comparison` (analyzes/compares fit results from both).

## Doc hygiene

On 2026-07-17, the implementation-log/plan/audit `.md` files scattered across
all three repos were consolidated into one `TODO.md` per repo. Several docs
in the other two repos turned out to describe work as "still open" or
"remaining" when it had actually already shipped — e.g. a whole "remaining
work" section in a `bias_model_comparison` plan doc, and a "stored fits are
stale, need refitting" note in a BBZ log that was already resolved. Both were
only caught by grepping the actual code/CSVs rather than trusting the doc.

**Takeaway:** before treating any "TODO" / "not yet implemented" / "remaining"
claim in a plan/log/audit doc (in any of the three repos) as still true, check
it against the current code or `git log` first — these docs go stale fast
relative to the code.

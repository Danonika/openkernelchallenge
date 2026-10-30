# Continuation (adapted from Auto-research-agent, MIT)

Continue from this session's saved evidence and handoff. Inspect the latest
experiment outcomes, explain failed expectations, choose the next justified
experiment and carry it through verification. Preserve prior candidate sources,
raw results and evaluation contracts. Review unfinished saved work before
duplicating it. An interrupted turn is neither validation nor rejection.

Respect remaining resources; GPU budget is cumulative across rounds. If progress
requires unavailable compute, a user answer or ungranted permission, describe
the blocker and return RESEARCH_STATUS: needs_input. Update RESEARCH_HANDOFF.md
before the final status line. Do not change the task or repeat exhausted ideas.

"""Package the already-tested pilot block as one uploadable Bash script."""
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / "PTFLOW_PILOT_200_COMMANDS.txt").read_text(encoding="utf-8")
text = text.replace(
    "# Paste this WHOLE block into the Palmetto login terminal. No upload needed.\n",
    "#!/usr/bin/env bash\n# Upload this file, then run: bash /scratch/$USER/ptflow/submit_pt_pilot200.sh\n",
    1,
)
needle = 'QUEUED=$(squeue -h -u "$USER" -o \'%i\')\n'
assert text.count(needle) == 1
text = text.replace(needle, needle + '''PILOT_JOBS=$(squeue -h -u "$USER" -n pt-B-pilot200 -o '%i')
if [[ -n "$PILOT_JOBS" ]]; then
    echo "A pilot is already queued/running: $PILOT_JOBS"
    echo "No duplicate job submitted. Check squeue before retrying."
    exit 1
fi
''', 1)
old_guard = '''if grep -Fxq "$OLD_JOB" <<< "$QUEUED"; then
    echo "The previous job $OLD_JOB is still queued/running. This pilot assumes it was cancelled."
    exit 1
fi
'''
assert text.count(old_guard) == 1
text = text.replace(old_guard, "", 1)
text = text.replace('OLD_JOB=$(cat "$SOURCE/job_id.txt")\n', "", 1)
text = text.replace(needle, "", 1)
text = text.replace("Cancelled run checkpoints:", "Source run checkpoints:")
text = text.replace("cancelled run files are preserved", "source run files are preserved")
path = root / "submit_pt_pilot200.sh"
path.write_text(text, encoding="utf-8", newline="\n")
assert b"\r" not in path.read_bytes()
print(path)

#!/bin/bash
# journal.sh — append a new entry to JOURNAL.md

TODAY=$(date +%Y-%m-%d)

echo -e "\n## $TODAY" >> JOURNAL.md
echo "### Goal" >> JOURNAL.md
echo "- $1" >> JOURNAL.md
echo -e "\n### What I did\n- " >> JOURNAL.md
echo -e "\n### Key learnings\n- " >> JOURNAL.md
echo -e "\n### Challenges\n- " >> JOURNAL.md
echo -e "\n### Next steps\n- " >> JOURNAL.md
echo "" >> JOURNAL.md

git add JOURNAL.md
git commit -m "docs(journal): add $TODAY entry"
git push

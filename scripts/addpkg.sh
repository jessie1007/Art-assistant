mkdir -p scripts
cat > scripts/addpkg.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: $0 pkg1 [pkg2 ...]"; exit 1
fi
pip install "$@"
pip freeze > requirements.txt
git add requirements.txt
echo "Installed: $*"
echo "Updated requirements.txt"
EOF
chmod +x scripts/addpkg.sh

#./scripts/addpkg.sh numpy pandas scikit-learn xxxx
# Check staged changes
# git status
# You'll see "requirements.txt" staged

# Commit
#git commit -m "chore: add matplotlib + pillow"
#git push

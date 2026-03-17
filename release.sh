#!/usr/bin/env bash
# release.sh — Build, tag, and publish a DragonchessAI release.
#
# Publishes to TWO repos:
#   origin   → joconno2/DragonchessAI          (game/platform: engine, plugin API, examples)
#   research → joconno2/DragonchessAI-Research  (research: training code, results, PLAN)
#
# Usage:
#   ./release.sh <version>              # e.g. ./release.sh v2.1.0
#   ./release.sh <version> --dry-run    # print steps without executing
#   ./release.sh <version> --no-build   # skip cmake build (use existing binary)
#
# Requires:
#   - GITHUB_TOKEN env var, or ~/gitToken file containing the token
#   - cmake, git, curl, jq

set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

VERSION="${1:-}"
DRY_RUN=false
SKIP_BUILD=false

for arg in "${@:2}"; do
    case "$arg" in
        --dry-run)   DRY_RUN=true ;;
        --no-build)  SKIP_BUILD=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if [[ -z "$VERSION" ]]; then
    echo "Usage: $0 <version> [--dry-run] [--no-build]"
    echo "Example: $0 v2.1.0"
    exit 1
fi

if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: version must be in the form vX.Y.Z (got: $VERSION)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# GitHub token
# ---------------------------------------------------------------------------

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    TOKEN_FILE="$HOME/gitToken"
    if [[ -f "$TOKEN_FILE" ]]; then
        GITHUB_TOKEN="$(cat "$TOKEN_FILE" | tr -d '[:space:]')"
    else
        echo "ERROR: No GitHub token found."
        echo "Set GITHUB_TOKEN env var or place token in ~/gitToken"
        exit 1
    fi
fi

PLATFORM_REPO="joconno2/DragonchessAI"
RESEARCH_REPO="joconno2/DragonchessAI-Research"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run() {
    echo "  + $*"
    if ! $DRY_RUN; then
        "$@"
    fi
}

info() { echo; echo "==> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

gh_release() {
    local REPO="$1" TAG="$2" NAME="$3" NOTES="$4"
    if $DRY_RUN; then
        echo "  [dry run] would POST release to https://api.github.com/repos/$REPO/releases"
        echo "DRYRUN"
        return
    fi
    local RESPONSE
    RESPONSE=$(curl -s -X POST \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        "https://api.github.com/repos/$REPO/releases" \
        -d "$(jq -n \
            --arg tag "$TAG" \
            --arg name "$NAME" \
            --arg body "$NOTES" \
            '{tag_name: $tag, name: $name, body: $body, draft: false, prerelease: false}')")
    local ID URL
    ID=$(echo "$RESPONSE" | jq -r '.id // empty')
    URL=$(echo "$RESPONSE" | jq -r '.html_url // empty')
    if [[ -z "$ID" ]]; then
        echo "WARNING: Failed to create release on $REPO:"
        echo "$RESPONSE" | jq .
        echo "FAILED"
        return
    fi
    echo "  release: $URL"
    echo "$ID"
}

upload_asset() {
    local REPO="$1" RELEASE_ID="$2" FILE="$3" CONTENT_TYPE="$4"
    local NAME
    NAME=$(basename "$FILE")
    $DRY_RUN && { echo "  [dry run] would upload $NAME to $REPO release $RELEASE_ID"; return; }
    curl -s -X POST \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Content-Type: $CONTENT_TYPE" \
        "https://uploads.github.com/repos/$REPO/releases/$RELEASE_ID/assets?name=$NAME" \
        --data-binary @"$FILE" | jq -r '.browser_download_url // "upload failed"'
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

info "Pre-flight checks"

command -v cmake >/dev/null || die "cmake not found"
command -v curl  >/dev/null || die "curl not found"
command -v jq    >/dev/null || die "jq not found"

git remote get-url origin   >/dev/null 2>&1 || die "remote 'origin' not configured"
git remote get-url research >/dev/null 2>&1 || die "remote 'research' not configured (run: git remote add research https://github.com/joconno2/DragonchessAI-Research.git)"

if git tag --list | grep -qx "$VERSION"; then
    die "Tag $VERSION already exists"
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "main" ]]; then
    echo "  WARNING: not on main branch (currently on: $BRANCH)"
    read -rp "  Release from this branch anyway? [y/N] " CONFIRM
    [[ "$CONFIRM" =~ ^[Yy]$ ]] || die "Aborted."
fi

echo "  version:   $VERSION"
echo "  platform:  $PLATFORM_REPO"
echo "  research:  $RESEARCH_REPO"
echo "  branch:    $BRANCH"
$DRY_RUN && echo "  [DRY RUN — no changes will be made]"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

if ! $SKIP_BUILD; then
    info "Building Linux binary"
    run cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    run cmake --build build --parallel "$(nproc)"
fi

[[ -f "build/dragonchess" ]] || die "Binary not found at build/dragonchess"

# ---------------------------------------------------------------------------
# Package dist
# ---------------------------------------------------------------------------

info "Packaging dist"

DIST_DIR="dist"
LINUX_DIR="$DIST_DIR/DragonchessAI-Linux-x64"
LINUX_TAR="$DIST_DIR/DragonchessAI-Linux-x64.tar.gz"

run mkdir -p "$LINUX_DIR"
run cp build/dragonchess "$LINUX_DIR/dragonchess"
run cp README.md LICENSE CHANGELOG.md "$LINUX_DIR/"
run cp -r examples "$LINUX_DIR/examples"

if ! $DRY_RUN; then
    (cd "$DIST_DIR" && tar -czf "DragonchessAI-Linux-x64.tar.gz" "DragonchessAI-Linux-x64/")
fi
echo "  packaged: $LINUX_TAR"

# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------

info "Staging changes"

# Platform files — clean engine, no research cruft
PLATFORM_FILES=(
    src/
    CMakeLists.txt
    README.md
    CHANGELOG.md
    LICENSE
    .gitignore
    assets/
    examples/
    setup/
)

# Research files — everything training/experiment related
RESEARCH_FILES=(
    train_td.py
    train_cc.py
    train_cma.py
    run_td_experiments.sh
    run_td_depth2_sweep.sh
    release.sh
    analyze_results.py
    evaluate_posttrain.py
    dashboard.py
    requirements.txt
    environment.yml
    PLAN.md
    cluster/
    figures/
    results/monolithic/
    results/cc/
)

ALL_FILES=("${PLATFORM_FILES[@]}" "${RESEARCH_FILES[@]}")

# Stage files that exist
for f in "${ALL_FILES[@]}"; do
    [[ -e "$f" ]] && git add "$f" 2>/dev/null || true
done

run git status --short

echo
read -rp "Commit message (default: 'Release $VERSION'): " MSG
MSG="${MSG:-Release $VERSION}"

run git commit -m "$MSG"

# ---------------------------------------------------------------------------
# Tag
# ---------------------------------------------------------------------------

info "Tagging $VERSION"

NOTES=$(awk "/^## \[${VERSION#v}\]/,/^## \[/" CHANGELOG.md 2>/dev/null \
    | grep -v "^## \[" | sed '/^[[:space:]]*$/d' \
    || echo "See CHANGELOG.md for details.")

run git tag -a "$VERSION" -m "Release $VERSION"

# ---------------------------------------------------------------------------
# Push — platform repo (origin)
# ---------------------------------------------------------------------------

info "Pushing platform repo (origin)"
run git push origin main
run git push origin "$VERSION"

# ---------------------------------------------------------------------------
# Push — research repo
# ---------------------------------------------------------------------------

info "Pushing research repo (research)"
run git push research main
run git push research "$VERSION"

# ---------------------------------------------------------------------------
# GitHub release — platform
# ---------------------------------------------------------------------------

info "Creating platform release ($PLATFORM_REPO)"
PLATFORM_RELEASE_ID=$(gh_release "$PLATFORM_REPO" "$VERSION" "DragonchessAI $VERSION" "$NOTES")

if [[ "$PLATFORM_RELEASE_ID" != "DRYRUN" && "$PLATFORM_RELEASE_ID" != "FAILED" ]]; then
    info "Uploading platform assets"
    upload_asset "$PLATFORM_REPO" "$PLATFORM_RELEASE_ID" "$LINUX_TAR" "application/gzip"
    WIN_ZIP="$DIST_DIR/DragonchessAI-Windows-x64.zip"
    [[ -f "$WIN_ZIP" ]] && upload_asset "$PLATFORM_REPO" "$PLATFORM_RELEASE_ID" "$WIN_ZIP" "application/zip"
fi

# ---------------------------------------------------------------------------
# GitHub release — research
# ---------------------------------------------------------------------------

info "Creating research release ($RESEARCH_REPO)"
RESEARCH_NOTES="Research snapshot $VERSION — see PLAN.md for experiment status."$'\n\n'"$NOTES"
RESEARCH_RELEASE_ID=$(gh_release "$RESEARCH_REPO" "$VERSION" "DragonchessAI-Research $VERSION" "$RESEARCH_NOTES")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo
echo "============================================"
echo " Released: DragonchessAI $VERSION"
echo "  platform: https://github.com/$PLATFORM_REPO/releases/tag/$VERSION"
echo "  research: https://github.com/$RESEARCH_REPO/releases/tag/$VERSION"
echo "============================================"

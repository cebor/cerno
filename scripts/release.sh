#!/usr/bin/env bash
# Release cerno: one version for every subproject, a changelog section, an annotated tag.
#
#     scripts/release.sh 0.2.0
#
# The changelog section is generated from the `Changelog:` trailers since the last tag, unless
# CHANGELOG.md already has a section for this version — then that one is used as written. So
# to edit the wording before releasing, generate it once, edit and commit it, and run again.
#
# Nothing is pushed. The last line printed is the push command.

set -euo pipefail

die() { echo "release: $*" >&2; exit 1; }

# The sed expressions below are GNU's (`0,/re/`, `-i` without a suffix). On macOS that is
# usually installed as gsed.
if sed --version >/dev/null 2>&1; then
    sed=sed
elif command -v gsed >/dev/null; then
    sed=gsed
else
    die "needs GNU sed; on macOS: brew install gnu-sed"
fi

cd "$(git rev-parse --show-toplevel)"

[[ $# -eq 1 ]] || die "usage: scripts/release.sh <major.minor.patch>"
version=$1
tag="v$version"
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "'$version' is not major.minor.patch"

[[ $(git symbolic-ref --short HEAD 2>/dev/null) == main ]] || die "not on main"
[[ -z $(git status --porcelain) ]] || die "the working tree is not clean"
! git rev-parse -q --verify "refs/tags/$tag" >/dev/null || die "$tag already exists"

last_tag=$(git tag --list 'v[0-9]*' --sort=-v:refname | head -n1)
if [[ -n $last_tag ]]; then
    [[ $(git rev-parse HEAD) != $(git rev-parse "$last_tag^{commit}") ]] ||
        die "nothing to release: HEAD is $last_tag"
    newest=$(printf '%s\n' "${last_tag#v}" "$version" | sort -V | tail -n1)
    [[ $newest == "$version" && ${last_tag#v} != "$version" ]] ||
        die "$version is not above the last release, $last_tag"
fi

# Every place the version is written down. They must agree before the release and after it.
versions() {
    echo "Cargo.toml $($sed -n '/^\[workspace\.package\]/,/^\[/ s/^version = "\(.*\)"/\1/p' Cargo.toml)"
    awk '/^name = "cerno-/ { name = $3; getline; gsub(/"/, "", $3); print "Cargo.lock:" name, $3 }' Cargo.lock
    $sed -n 's/^\(cerno-[a-z]*\) = { path = .*, version = "=\(.*\)" }$/Cargo.toml:\1 \2/p' Cargo.toml
    echo "pyproject.toml $($sed -n '0,/^version = / s/^version = "\(.*\)"/\1/p' sdks/python/pyproject.toml)"
    echo "__init__.py $($sed -n 's/^__version__ = "\(.*\)"/\1/p' sdks/python/src/cerno/__init__.py)"
    awk '/^name = "cerno-sdk"$/ { getline; gsub(/"/, "", $3); print "uv.lock", $3 }' sdks/python/uv.lock
    node -e '
        const pkg = require("./sdks/typescript/package.json");
        const lock = require("./sdks/typescript/package-lock.json");
        const spec = require("./spec/openapi.json");
        console.log("package.json", pkg.version);
        console.log("package-lock.json", lock.version);
        console.log("package-lock.json:packages", lock.packages[""].version);
        console.log("openapi.json", spec.info.version);
    '
}

# Fails, listing every place, unless all of them hold exactly $1.
all_at() {
    local listing
    listing=$(versions)
    if awk -v want="$1" '$2 != want { bad = 1 } END { exit !bad }' <<<"$listing"; then
        echo "$listing" >&2
        return 1
    fi
}

current=$($sed -n '/^\[workspace\.package\]/,/^\[/ s/^version = "\(.*\)"/\1/p' Cargo.toml)
all_at "$current" || die "the versions above disagree; fix that by hand first"

# --- The changelog section ----------------------------------------------------------------

# Prints the body of CHANGELOG.md's section for $1, without its heading.
section_of() {
    awk -v head="## [$1]" '
        index($0, head) == 1 { inside = 1; next }
        inside && /^## \[/ { exit }
        inside { print }
    ' CHANGELOG.md
}

readonly all_scopes="service rust python typescript tui"

# Prints the scopes whose files commit $1 touches, one per line. Docs, CI, the benchmark and
# the lockfiles belong to none.
scopes_of() {
    git show --name-only --format= "$1" | while read -r path; do
        case $path in
            crates/cerno-server/* | crates/cerno-core/* | crates/cerno-host/* | \
                crates/cerno-types/* | spec/*) echo service ;;
            crates/cerno-sdk/*) echo rust ;;
            sdks/python/*) echo python ;;
            sdks/typescript/*) echo typescript ;;
            crates/cerno-tui/*) echo tui ;;
        esac
    done
}

# Turns scopes, one per line and in any order, into "Service, Python" in the fixed order.
scope_names() {
    local found=$1 scope names=""
    for scope in $all_scopes; do
        grep -qx "$scope" <<<"$found" || continue
        case $scope in
            service) scope=Service ;;
            rust) scope=Rust ;;
            python) scope=Python ;;
            typescript) scope=TypeScript ;;
            tui) scope=TUI ;;
        esac
        names+="${names:+, }$scope"
    done
    echo "$names"
}

if grep -qF "## [$version]" CHANGELOG.md; then
    echo "Using the section for $version already in CHANGELOG.md."
else
    entries=""
    range=${last_tag:+$last_tag..}HEAD
    while IFS=$'\x1f' read -r -d $'\x1e' hash subject kinds scope_trailer; do
        hash=${hash//$'\n'/}
        kinds=${kinds//$'\n'/}
        scope_trailer=${scope_trailer//$'\n'/}
        [[ -n $kinds ]] || continue
        [[ $kinds != *,* ]] || die "$hash has more than one Changelog trailer: $kinds"
        case $kinds in
            added | changed | deprecated | removed | fixed | security | performance) ;;
            *) die "$hash has an unknown Changelog trailer '$kinds' (see CONTRIBUTING.md)" ;;
        esac

        # Scope overrides the paths, for a change that only adjusts its neighbours.
        if [[ -n $scope_trailer ]]; then
            scopes=$(tr ',' '\n' <<<"$scope_trailer" | tr -d ' \t' | tr '[:upper:]' '[:lower:]')
            while read -r scope; do
                [[ " $all_scopes " == *" $scope "* ]] ||
                    die "$hash has an unknown Scope '$scope' (one of: $all_scopes)"
            done <<<"$scopes"
        else
            scopes=$(scopes_of "$hash")
            [[ -n $scopes ]] ||
                die "$hash touches no subproject; add a Scope trailer (one of: $all_scopes)"
        fi
        entries+="$kinds - **$(scope_names "$scopes"):** $subject ($hash)"$'\n'
    done < <(git log --reverse --format='%h%x1f%s%x1f%(trailers:key=Changelog,valueonly,separator=%x2C)%x1f%(trailers:key=Scope,valueonly,separator=%x2C)%x1e' "$range")

    section="## [$version] - $(date +%F)"$'\n'
    for kind in Added Changed Deprecated Removed Fixed Security Performance; do
        lower=$(tr '[:upper:]' '[:lower:]' <<<"$kind")
        lines=$(awk -v k="$lower" '$1 == k { sub(/^[^ ]+ /, ""); print }' <<<"$entries")
        [[ -n $lines ]] || continue
        section+=$'\n'"### $kind"$'\n\n'"$lines"$'\n'
    done
    [[ $section == *'### '* ]] || die "nothing user-visible since ${last_tag:-the first commit}"

    # Newest first: in front of the first existing section, or at the end if there is none.
    section_file=$(mktemp)
    trap 'rm -f "$section_file"' EXIT
    printf '%s\n' "$section" >"$section_file"
    awk -v file="$section_file" '
        !done && /^## \[/ { while ((getline line < file) > 0) print line; done = 1 }
        { print }
        END { if (!done) { print ""; while ((getline line < file) > 0) print line } }
    ' CHANGELOG.md >CHANGELOG.md.new
    mv CHANGELOG.md.new CHANGELOG.md
    echo "Wrote the section for $version into CHANGELOG.md."
fi

# --- The version bump ---------------------------------------------------------------------

if [[ $version != "$current" ]]; then
    trap 'echo "release: failed while bumping; git restore . puts everything back" >&2' ERR
    $sed -i "/^\[workspace\.package\]/,/^\[/ s/^version = \".*\"/version = \"$version\"/" Cargo.toml
    $sed -i "s/^\(cerno-[a-z]* = { path = .*, version = \"=\).*\(\" }\)$/\1$version\2/" Cargo.toml
    $sed -i "0,/^version = / s/^version = \".*\"/version = \"$version\"/" sdks/python/pyproject.toml
    $sed -i "s/^__version__ = \".*\"/__version__ = \"$version\"/" sdks/python/src/cerno/__init__.py
    # Not `npm version`: it reformats package.json, expanding every inline array and object.
    $sed -i "0,/^  \"version\": / s/^  \"version\": \".*\"/  \"version\": \"$version\"/" sdks/typescript/package.json
    node -e '
        const fs = require("fs");
        const path = "sdks/typescript/package-lock.json";
        const lock = JSON.parse(fs.readFileSync(path, "utf8"));
        lock.version = lock.packages[""].version = process.argv[1];
        fs.writeFileSync(path, JSON.stringify(lock, null, 2) + "\n");
    ' "$version"
    (cd sdks/python && uv lock --quiet)
    cargo update --workspace --quiet
    # utoipa takes info.version from CARGO_PKG_VERSION, so the spec follows the bump.
    cargo run --quiet -p cerno-server --bin cerno-openapi >spec/openapi.json
    all_at "$version" || die "the bump missed the places above"
fi

if [[ -n $(git status --porcelain) ]]; then
    git add -A
    git commit --quiet -m "Release $version"
    echo "Committed Release $version."
fi

# --cleanup=verbatim, because the default would strip the ### headings as comments.
git tag -a "$tag" --cleanup=verbatim -F - <<EOF
cerno $version
$(section_of "$version")
EOF
echo "Tagged $tag."
echo
echo "    git push origin main $tag && git push github main $tag"

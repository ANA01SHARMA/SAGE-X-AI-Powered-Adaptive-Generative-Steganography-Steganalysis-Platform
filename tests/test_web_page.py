"""Static checks on the single-page frontend.

The page has no build step and no JS test runner, so a renamed element id fails
silently in the browser: `$("...")` returns null, the handler throws, and the UI simply
stops responding with nothing in the terminal to show for it. These tests catch that
class of mistake, which has bitten this file once already.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "sagex" / "web" / "static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def declared_ids(html: str) -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', html))


@pytest.fixture(scope="module")
def referenced_ids(html: str) -> set[str]:
    """Every id the script looks up, via the `$` helper or getElementById."""
    return set(re.findall(r'\$\(\s*"([^"]+)"\s*\)', html)) | set(
        re.findall(r'getElementById\(\s*"([^"]+)"\s*\)', html)
    )


def test_every_referenced_id_exists(referenced_ids, declared_ids):
    """A lookup for an element that is not in the markup returns null and throws."""
    missing = sorted(referenced_ids - declared_ids)
    assert not missing, (
        f"the script looks up ids that the markup does not define: {missing}. "
        "In the browser this throws and the handler stops silently."
    )


def test_no_orphan_ids(declared_ids, referenced_ids, html):
    """Flag ids nothing uses, which usually means a rename was left half-finished."""
    # Some ids are reached by constructed names rather than a literal lookup, so a
    # regex cannot see the reference:
    #   "tab-" + dataset.tab   in the nav handler
    #   inputId + "-drop"      in dropZone()
    conventional = {
        i for i in declared_ids
        if i.startswith("tab-") or (i.endswith("-drop") and i[: -len("-drop")] in referenced_ids)
    }
    orphans = sorted(declared_ids - referenced_ids - conventional)
    assert not orphans, f"these ids are defined but never used: {orphans}"


def test_every_tab_button_has_a_panel(html, declared_ids):
    """Each nav button must point at a section that exists."""
    tabs = re.findall(r'data-tab="([^"]+)"', html)
    assert tabs, "no tab buttons found"
    for tab in tabs:
        assert f"tab-{tab}" in declared_ids, f"nav button '{tab}' has no matching panel"


def test_no_external_resources(html):
    """The page must work offline: everything inline, nothing fetched from a CDN."""
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    assert not external, f"page pulls in external resources: {external}"


def test_api_paths_match_the_backend(html):
    """Every endpoint the page calls must be a route the server actually serves."""
    from sagex.web.app import app

    called = set(re.findall(r'"(/api/[a-z_]+)"', html))
    served = {route.path for route in app.routes}
    missing = sorted(called - served)
    assert not missing, f"page calls endpoints the backend does not serve: {missing}"

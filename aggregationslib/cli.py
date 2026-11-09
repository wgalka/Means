"""Command line entry point for the Shiny interface."""

import argparse
from importlib import invalidate_caches
from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path
import site
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement


def _missing_app_requirements():
    try:
        requirements = requires("aggregationslib") or []
    except PackageNotFoundError:
        # Support running the CLI as a module from a source checkout.
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject.open("rb") as file:
            extras = tomllib.load(file)["project"]["optional-dependencies"]["app"]
        requirements = [f'{item}; extra == "app"' for item in extras]

    missing = []
    for item in requirements:
        requirement = Requirement(item)
        marker = requirement.marker
        if not marker or not marker.evaluate({"extra": "app"}) or marker.evaluate({"extra": ""}):
            continue
        try:
            if requirement.specifier.contains(version(requirement.name), prereleases=True):
                continue
        except PackageNotFoundError:
            pass
        requirement.marker = None
        missing.append(str(requirement))
    return missing


def main():
    parser = argparse.ArgumentParser(prog="agglib")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("start", aliases=["serve"], help="Open the local Shiny app")
    parser.parse_args()

    missing = _missing_app_requirements()
    if missing:
        if not sys.stdin.isatty():
            parser.exit(
                1,
                'Run agglib start in a terminal or install the app with '
                'python -m pip install "aggregationslib[app]".\n',
            )
        try:
            answer = input("Install the graphical interface? [Y/n] ").strip().lower()
        except EOFError:
            return
        except KeyboardInterrupt:
            parser.exit(130, "\n")
        if answer not in {"", "y", "yes"}:
            return

        try:
            subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
        except subprocess.CalledProcessError as exc:
            parser.exit(exc.returncode, "Could not install the graphical interface dependencies.\n")
        if site.ENABLE_USER_SITE:
            user_site = site.getusersitepackages()
            if user_site not in sys.path and Path(user_site).is_dir():
                site.addsitedir(user_site)
        invalidate_caches()

    from shiny import run_app
    from apps.shiny.app import app

    run_app(app, host="127.0.0.1", launch_browser=True)


if __name__ == "__main__":
    main()

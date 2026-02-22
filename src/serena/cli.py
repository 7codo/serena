import os
import sys
from collections.abc import Iterator
from logging import Logger
from pathlib import Path
from typing import Any, Literal

import click
from sensai.util import logging


from serena.constants import (
    TOOL_TIMEOUT,
    TRACE_LSP_COMMUNICATION,
    SERENA_LOG_FORMAT,
)
from serena.mcp import SerenaMCPFactory

log = logging.getLogger(__name__)


def find_project_root(root: str | Path | None = None) -> str | None:
    """Find project root by walking up from CWD.

    Checks for .serena/project.yml first (explicit Serena project), then .git (git root).

    :param root: If provided, constrains the search to this directory and below
                 (acts as a virtual filesystem root). Search stops at this boundary.
    :return: absolute path to project root or None if not suitable root is found
    """
    current = Path.cwd().resolve()
    boundary = Path(root).resolve() if root is not None else None

    def ancestors() -> Iterator[Path]:
        """Yield current directory and ancestors up to boundary."""
        yield current
        for parent in current.parents:
            yield parent
            if boundary is not None and parent == boundary:
                return

    # First pass: look for .serena
    for directory in ancestors():
        if (directory / ".serena" / "project.yml").is_file():
            return str(directory)

    # Second pass: look for .git
    for directory in ancestors():
        if (directory / ".git").exists():  # .git can be file (worktree) or dir
            return str(directory)

    return None


# --------------------- Utilities -------------------------------------


class ProjectType(click.ParamType):
    """ParamType allowing either a project name or a path to a project directory."""

    name = "[PROJECT_NAME|PROJECT_PATH]"

    def convert(self, value: str, param: Any, ctx: Any) -> str:
        path = Path(value).resolve()
        if path.exists() and path.is_dir():
            return str(path)
        return value


PROJECT_TYPE = ProjectType()


class AutoRegisteringGroup(click.Group):
    """
    A click.Group subclass that automatically registers any click.Command
    attributes defined on the class into the group.

    After initialization, it inspects its own class for attributes that are
    instances of click.Command (typically created via @click.command) and
    calls self.add_command(cmd) on each. This lets you define your commands
    as static methods on the subclass for IDE-friendly organization without
    manual registration.
    """

    def __init__(self, name: str, help: str):
        super().__init__(name=name, help=help)
        # Scan class attributes for click.Command instances and register them.
        for attr in dir(self.__class__):
            cmd = getattr(self.__class__, attr)
            if isinstance(cmd, click.Command):
                self.add_command(cmd)


class TopLevelCommands(AutoRegisteringGroup):
    """Root CLI group containing the core Serena commands."""

    def __init__(self) -> None:
        super().__init__(name="serena", help="Serena CLI commands. You can run `<command> --help` for more info on each command.")

    @staticmethod
    @click.command("start-mcp-server", help="Starts the Serena MCP server.")
    @click.option("--project", "project", type=PROJECT_TYPE, default=None, help="Path or name of project to activate at startup.")
    @click.option(
        "--transport",
        type=click.Choice(["stdio", "sse", "streamable-http"]),
        default="stdio",
        show_default=True,
        help="Transport protocol.",
    )
    @click.option(
        "--host",
        type=str,
        default="0.0.0.0",
        show_default=True,
        help="Listen address for the MCP server (when using corresponding transport).",
    )
    @click.option(
        "--port", type=int, default=8000, show_default=True, help="Listen port for the MCP server (when using corresponding transport)."
    )
    @click.option(
        "--log-level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        default=None,
        help="Override log level in config.",
    )
    def start_mcp_server(
        project: str | None,
        transport: Literal["stdio", "sse", "streamable-http"],
        host: str,
        port: int,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None,
    ) -> None:
        # initialize logging, using INFO level initially (will later be adjusted by SerenaAgent according to the config)
        #   * memory log handler (for use by GUI/Dashboard)
        #   * stream handler for stderr (for direct console output, which will also be captured by clients like Claude Desktop)
        #   * file handler
        # (Note that stdout must never be used for logging, as it is used by the MCP server to communicate with the client.)
        Logger.root.setLevel(logging.INFO)
        formatter = logging.Formatter(SERENA_LOG_FORMAT)
        
        stderr_handler = logging.StreamHandler(stream=sys.stderr)
        stderr_handler.formatter = formatter
        Logger.root.addHandler(stderr_handler)
        
        # file_handler = logging.FileHandler(log_path, mode="w")
        # file_handler.formatter = formatter
        # Logger.root.addHandler(file_handler)

        log.info("Initializing Serena MCP server")
        # log.info("Storing logs in %s", log_path)

        
        factory = SerenaMCPFactory(project=project)
        server = factory.create_mcp_server(
            host=host,
            port=port,
            log_level=log_level,
            trace_lsp_communication=TRACE_LSP_COMMUNICATION,
            tool_timeout=TOOL_TIMEOUT,
        )
        
        log.info("Starting MCP server …")
        server.run(transport=transport)







# Expose groups so we can reference them in pyproject.toml

# Expose toplevel commands for the same reason
top_level = TopLevelCommands()
start_mcp_server = top_level.start_mcp_server


def get_help() -> str:
    """Retrieve the help text for the top-level Serena CLI."""
    return top_level.get_help(click.Context(top_level, info_name="serena"))
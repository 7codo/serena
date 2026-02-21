"""
The Serena Model Context Protocol (MCP) Server
"""

import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import docstring_parser
from mcp.server.fastmcp import server
from mcp.server.fastmcp.server import FastMCP, Settings
from mcp.server.fastmcp.tools.base import Tool as MCPTool
from mcp.types import ToolAnnotations
from pydantic_settings import SettingsConfigDict
from sensai.util import logging

from serena.agent import (
    SerenaAgent,
)
from serena.constants import SERENA_LOG_FORMAT
from serena.tools import Tool
from serena.util.exception import show_fatal_exception_safe

log = logging.getLogger(__name__)


def configure_logging(*args, **kwargs) -> None:  # type: ignore
    # We only do something here if logging has not yet been configured.
    # Normally, logging is configured in the MCP server startup script.
    if not logging.is_enabled():
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, format=SERENA_LOG_FORMAT)


# patch the logging configuration function in fastmcp, because it's hard-coded and broken
server.configure_logging = configure_logging  # type: ignore


@dataclass
class SerenaMCPRequestContext:
    agent: SerenaAgent


class SerenaMCPFactory:
    """
    Factory for the creation of the Serena MCP server with an associated SerenaAgent.
    """

    def __init__(self, project: str | None = None):
        """
        :param context: The context name or path to context file
        :param project: Either an absolute path to the project directory or a name of an already registered project.
            If the project passed here hasn't been registered yet, it will be registered automatically and can be activated by its name
            afterward.
        """
        self.project = project
        self.agent: SerenaAgent | None = None


    @staticmethod
    def make_mcp_tool(tool: Tool) -> MCPTool:
        """
        Create an MCP tool from a Serena Tool instance.

        :param tool: The Serena Tool instance to convert.
        
        """
        func_name = tool.get_name()
        func_doc = tool.get_apply_docstring() or ""
        func_arg_metadata = tool.get_apply_fn_metadata()
        is_async = False
        parameters = func_arg_metadata.arg_model.model_json_schema()

        docstring = docstring_parser.parse(func_doc)

        # Mount the tool description as a combination of the docstring description and
        # the return value description, if it exists.
        if docstring.description:
            func_doc = docstring.description
        else:
            func_doc = ""
        func_doc = func_doc.strip().strip(".")
        if func_doc:
            func_doc += "."
        if docstring.returns and (docstring_returns_descr := docstring.returns.description):
            # Only add a space before "Returns" if func_doc is not empty
            prefix = " " if func_doc else ""
            func_doc = f"{func_doc}{prefix}Returns {docstring_returns_descr.strip().strip('.')}."

        # Parse the parameter descriptions from the docstring and add pass its description
        # to the parameter schema.
        docstring_params = {param.arg_name: param for param in docstring.params}
        parameters_properties: dict[str, dict[str, Any]] = parameters["properties"]
        for parameter, properties in parameters_properties.items():
            if (param_doc := docstring_params.get(parameter)) and param_doc.description:
                param_desc = f"{param_doc.description.strip().strip('.') + '.'}"
                properties["description"] = param_desc[0].upper() + param_desc[1:]

        def execute_fn(**kwargs) -> str:  # type: ignore
            return tool.apply_ex(log_call=True, catch_exceptions=True, **kwargs)

        # Generate human-readable title from snake_case tool name
        tool_title = " ".join(word.capitalize() for word in func_name.split("_"))

        # Create annotations with appropriate hints based on tool capabilities
        can_edit = tool.can_edit()
        annotations = ToolAnnotations(
            title=tool_title,
            readOnlyHint=not can_edit,
            destructiveHint=can_edit,
        )

        return MCPTool(
            fn=execute_fn,
            name=func_name,
            description=func_doc,
            parameters=parameters,
            fn_metadata=func_arg_metadata,
            is_async=is_async,
            # keep the value in sync with the kwarg name in Tool.apply_ex. The mcp sdk uses reflection to infer this
            # when the tool is constructed via from_function (which is a bit crazy IMO, but well...)
            context_kwarg="mcp_ctx",
            annotations=annotations,
            title=tool_title,
        )

    def _iter_tools(self) -> Iterator[Tool]:
        assert self.agent is not None
        yield from self.agent.get_exposed_tool_instances()

    # noinspection PyProtectedMember
    def _set_mcp_tools(self, mcp: FastMCP) -> None:
        """Update the tools in the MCP server"""
        if mcp is not None:
            mcp._tool_manager._tools = {}
            for tool in self._iter_tools():
                mcp_tool = self.make_mcp_tool(tool)
                mcp._tool_manager._tools[tool.get_name()] = mcp_tool
            log.info(f"Starting MCP server with {len(mcp._tool_manager._tools)} tools: {list(mcp._tool_manager._tools.keys())}")

    def _create_serena_agent(self) -> SerenaAgent:
        return SerenaAgent(
            project=self.project
        )

    def create_mcp_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None,
        trace_lsp_communication: bool | None = None,
        tool_timeout: float | None = None,
    ) -> FastMCP:
        """
        Create an MCP server with process-isolated SerenaAgent to prevent asyncio contamination.

        :param host: The host to bind to
        :param port: The port to bind to
        :param modes: List of mode names or paths to mode files
        :param language_backend: the language backend to use, overriding the configuration setting.
        :param enable_web_dashboard: Whether to enable the web dashboard. If not specified, will take the value from the serena configuration.
        :param enable_gui_log_window: Whether to enable the GUI log window. It currently does not work on macOS, and setting this to True will be ignored then.
            If not specified, will take the value from the serena configuration.
        :param open_web_dashboard: Whether to open the web dashboard on launch.
            If not specified, will take the value from the serena configuration.
        :param log_level: Log level. If not specified, will take the value from the serena configuration.
        :param trace_lsp_communication: Whether to trace the communication between Serena and the language servers.
            This is useful for debugging language server issues.
        :param tool_timeout: Timeout in seconds for tool execution. If not specified, will take the value from the serena configuration.
        """
        try:

            
            self.agent = self._create_serena_agent()

        except Exception as e:
            show_fatal_exception_safe(e)
            raise

        # Override model_config to disable the use of `.env` files for reading settings, because user projects are likely to contain
        # `.env` files (e.g. containing LOG_LEVEL) that are not supposed to override the MCP settings;
        # retain only FASTMCP_ prefix for already set environment variables.
        Settings.model_config = SettingsConfigDict(env_prefix="FASTMCP_")
        mcp = FastMCP(lifespan=self.server_lifespan, host=host, port=port)
        return mcp

    @asynccontextmanager
    async def server_lifespan(self, mcp_server: FastMCP) -> AsyncIterator[None]:
        """Manage server startup and shutdown lifecycle."""
        
        self._set_mcp_tools(mcp_server)
        log.info("MCP server lifetime setup complete")
        yield
        log.info("MCP server shutting down")

